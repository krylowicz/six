import warnings
import numpy as np
import pandas as pd
from packaging import version

import torch
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer

from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.optimizers.optimizer import DPOptimizer

class Discriminator(Module):
  def __init__(self, input_dim, discriminator_dim, pac=10):
    super().__init__()
    dim = input_dim * pac
    self.pac = pac
    self.pacdim = dim
    seq = []
    for item in list(discriminator_dim):
      seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
      dim = item

    seq += [Linear(dim, 1)]
    self.seq = Sequential(*seq)

  def forward(self, input_):
    assert input_.size()[0] % self.pac == 0
    return self.seq(input_.view(-1, self.pacdim))

class Residual(Module):
  def __init__(self, i, o):
    super().__init__()
    self.fc = Linear(i, o)
    self.bn = BatchNorm1d(o)
    self.relu = ReLU()

  def forward(self, input_):
    out = self.fc(input_)
    out = self.bn(out)
    out = self.relu(out)
    return torch.cat([out, input_], dim=1)

class Generator(Module):
  def __init__(self, embedding_dim, generator_dim, data_dim):
    super().__init__()
    dim = embedding_dim
    seq = []
    for item in list(generator_dim):
      seq += [Residual(dim, item)]
      dim += item
    seq.append(Linear(dim, data_dim))
    self.seq = Sequential(*seq)

  def forward(self, input_):
    return self.seq(input_)

class CTGANSynthesizer(BaseSynthesizer):
  def __init__(
    self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
    generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
    discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
    log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True
  ):
    assert batch_size % 2 == 0

    self._embedding_dim = embedding_dim
    self._generator_dim = generator_dim
    self._discriminator_dim = discriminator_dim

    self._generator_lr = generator_lr
    self._generator_decay = generator_decay
    self._discriminator_lr = discriminator_lr
    self._discriminator_decay = discriminator_decay

    self._batch_size = batch_size
    self._discriminator_steps = discriminator_steps
    self._log_frequency = log_frequency
    self._verbose = verbose
    self._epochs = epochs
    self.pac = pac

    if not cuda or not torch.cuda.is_available():
      device = 'cpu'
    elif isinstance(cuda, str):
      device = cuda
    else:
      device = 'cuda'

    self._device = torch.device(device)

    self._transformer = None
    self._data_sampler = None
    self._generator = None

  @staticmethod
  def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    if version.parse(torch.__version__) < version.parse('1.2.0'):
      for _ in range(10):
        transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
        if not torch.isnan(transformed).any():
            return transformed
      raise ValueError('gumbel_softmax returning NaN.')

    return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
  
  def _apply_activate(self, data):
    data_t = []
    st = 0
    for column_info in self._transformer.output_info_list:
      for span_info in column_info:
        if span_info.activation_fn == 'tanh':
          ed = st + span_info.dim
          data_t.append(torch.tanh(data[:, st:ed]))
          st = ed
        elif span_info.activation_fn == 'softmax':
          ed = st + span_info.dim
          transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
          data_t.append(transformed)
          st = ed
        else:
          raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

    return torch.cat(data_t, dim=1)

  def _cond_loss(self, data, c, m):
    loss = []
    st = 0
    st_c = 0
    for column_info in self._transformer.output_info_list:
      for span_info in column_info:
        if len(column_info) != 1 or span_info.activation_fn != 'softmax':
          # not discrete column
          st += span_info.dim
        else:
          ed = st + span_info.dim
          ed_c = st_c + span_info.dim
          tmp = functional.cross_entropy(
            data[:, st:ed],
            torch.argmax(c[:, st_c:ed_c], dim=1),
            reduction='none'
          )
          loss.append(tmp)
          st = ed
          st_c = ed_c

    loss = torch.stack(loss, dim=1)

    return (loss * m).sum() / data.size()[0]

  def _validate_discrete_columns(self, train_data, discrete_columns):
    if isinstance(train_data, pd.DataFrame):
      invalid_columns = set(discrete_columns) - set(train_data.columns)
    elif isinstance(train_data, np.ndarray):
      invalid_columns = [
          column for column in discrete_columns
          if column < 0 or column >= train_data.shape[1]
      ]
    else:
      raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

    if invalid_columns:
      raise ValueError(f'Invalid columns found: {invalid_columns}')
      
  def fit(self, train_data, discrete_columns=(), epochs=None, dp=False):
    if epochs is None:
      epochs = self._epochs
    else:
      warnings.warn(
        ('`epochs` argument in `fit` method has been deprecated and will be removed '
        'in a future version. Please pass `epochs` to the constructor instead'),
        DeprecationWarning
      )

    self._transformer = DataTransformer()
    self._transformer.fit(train_data, discrete_columns)

    train_data = self._transformer.transform(train_data)

    self._data_sampler = DataSampler(
      train_data,
      self._transformer.output_info_list,
      self._log_frequency)

    data_dim = self._transformer.output_dimensions

    self._generator = Generator(
      self._embedding_dim + self._data_sampler.dim_cond_vec(),
      self._generator_dim,
      data_dim
    ).to(self._device)

    discriminator = Discriminator(
      data_dim + self._data_sampler.dim_cond_vec(),
      self._discriminator_dim,
      pac=self.pac
    ).to(self._device)

    optimizerG = optim.Adam(self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9), weight_decay=self._generator_decay)
    optimizerD = optim.Adam(discriminator.parameters(), lr=self._discriminator_lr, betas=(0.5, 0.9), weight_decay=self._discriminator_decay)

    if dp: 
      sample_rate = self._batch_size / len(train_data)
      expected_batch_size = int(len(train_data) * sample_rate)

      alphas = [0.01 + x / 5.0 for x in range(1, 10)]
    
      accountant = RDPAccountant()
      discriminator = GradSampleModule(discriminator)
      optimizerD = DPOptimizer(
        optimizer=optimizerD,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        expected_batch_size=expected_batch_size,
      )

      optimizerD.attach_step_hook(
        accountant.get_optimizer_hook_fn(
          sample_rate=sample_rate,
        )
      )

    mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
    std = mean + 1

    steps_per_epoch = max(len(train_data) // self._batch_size, 1)
    for i in range(epochs):
      for _ in range(steps_per_epoch):
        for _ in range(self._discriminator_steps):
          fakez = torch.normal(mean=mean, std=std)

          condvec = self._data_sampler.sample_condvec(self._batch_size)
          if condvec is None:
            c1, m1, col, opt = None, None, None, None
            real = self._data_sampler.sample_data(self._batch_size, col, opt)
          else:
            c1, m1, col, opt = condvec
            c1 = torch.from_numpy(c1).to(self._device)
            m1 = torch.from_numpy(m1).to(self._device)
            fakez = torch.cat([fakez, c1], dim=1)

            perm = np.arange(self._batch_size)
            np.random.shuffle(perm)
            real = self._data_sampler.sample_data(
                self._batch_size, col[perm], opt[perm])
            c2 = c1[perm]

          fake = self._generator(fakez)
          fakeact = self._apply_activate(fake)

          real = torch.from_numpy(real.astype('float32')).to(self._device)

          if c1 is not None:
            fake_cat = torch.cat([fakeact, c1], dim=1)
            real_cat = torch.cat([real, c2], dim=1)
          else:
            real_cat = real
            fake_cat = fakeact

          y_fake = discriminator(fake_cat)
          y_real = discriminator(real_cat)

          # pen = discriminator._module.calc_gradient_penalty(
          #   real_cat, fake_cat, self._device, self.pac)
          loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

          optimizerD.zero_grad()
          # pen.backward(retain_graph=True)
          loss_d.backward()
          optimizerD.step()

      fakez = torch.normal(mean=mean, std=std)
      condvec = self._data_sampler.sample_condvec(self._batch_size)

      if condvec is None:
        c1, m1, col, opt = None, None, None, None
      else:
        c1, m1, col, opt = condvec
        c1 = torch.from_numpy(c1).to(self._device)
        m1 = torch.from_numpy(m1).to(self._device)
        fakez = torch.cat([fakez, c1], dim=1)

      fake = self._generator(fakez)
      fakeact = self._apply_activate(fake)

      if c1 is not None:
        y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
      else:
        y_fake = discriminator(fakeact)

      cross_entropy = 0 if condvec is None else self._cond_loss(fake, c1, m1)
      loss_g = -torch.mean(y_fake) + cross_entropy

      optimizerG.zero_grad()
      loss_g.backward()
      optimizerG.step()

      if self._verbose:
        print(
          f"epoch: {i+1}, loss_g: {loss_g.detach().cpu(): .4f}, loss_d: {loss_d.detach().cpu(): .4f}", flush=True
        )

        if dp:
          delta = 1.0
          epsilon, best_alpha = accountant.get_privacy_spent(delta=delta, alphas=alphas)
          print(f"(ε = {epsilon}, δ = {delta}) for α = {best_alpha}")

  def sample(self, n, condition_column=None, condition_value=None):
    if condition_column is not None and condition_value is not None:
      condition_info = self._transformer.convert_column_name_value_to_id(
        condition_column, condition_value)
      global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
        condition_info, self._batch_size)
    else:
      global_condition_vec = None

    steps = n // self._batch_size + 1
    data = []
    for _ in range(steps):
      mean = torch.zeros(self._batch_size, self._embedding_dim)
      std = mean + 1
      fakez = torch.normal(mean=mean, std=std).to(self._device)

      if global_condition_vec is not None:
          condvec = global_condition_vec.copy()
      else:
          condvec = self._data_sampler.sample_original_condvec(self._batch_size)

      if condvec is not None:
        c1 = condvec
        c1 = torch.from_numpy(c1).to(self._device)
        fakez = torch.cat([fakez, c1], dim=1)

      fake = self._generator(fakez)
      fakeact = self._apply_activate(fake)
      data.append(fakeact.detach().cpu().numpy())

    data = np.concatenate(data, axis=0)
    data = data[:n]

    return self._transformer.inverse_transform(data)

  def set_device(self, device):
    self._device = device
    if self._generator is not None:
      self._generator.to(self._device)
