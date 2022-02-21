import numpy as np
from ctgan_synthesizer import CTGANSynthesizer
from sdv.tabular.base import BaseTabularModel

class CTGANModel(BaseTabularModel):
  _MODEL_CLASS = None
  _model_kwargs = None
  
  _DTYPE_TRANSFORMERS = {
    '0': None
  }

  def _build_model(self):
    return self._MODEL_CLASS(**self._model_kwargs)

  def _fit(self, table_data):
    self._model = self._build_model()

    categoricals = []
    fields_before_transform = self._metadata.get_fields()
    for field in table_data.columns:
      if field in fields_before_transform:
        meta = fields_before_transform[field]
        if meta['type'] == 'categorical':
          categoricals.append(field)
      else:
        field_data = table_data[field].dropna()
        if set(field_data.unique()) == {0.0, 1.0}:
          # booleans encoded as float values must be modeled as bool
          field_data = field_data.astype(bool)

        dtype = field_data.infer_objects().dtype
        try:
          kind = np.dtype(dtype).kind
        except TypeError:
          # probably category
          kind = 'O'
        if kind in {'O', 'b'}:
          categoricals.append(field)

    self._model.fit(table_data, discrete_columns=categoricals)
  
  def _sample(self, num_rows, conditions=None):
    if conditions is None:
      return self._model.sample(num_rows)
    raise NotImplementedError(f"{self._MODEL_CLASS} doesn't support conditional sampling.")

class CTGAN(CTGANModel):
  _MODEL_CLASS = CTGANSynthesizer

  def __init__(
    self, field_names=None, field_types=None, field_transformers=None,
    anonymize_fields=None, primary_key=None, constraints=None, table_metadata=None,
    embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
    generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
    discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
    log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True,
    rounding='auto', min_value='auto', max_value='auto'
  ):
    super().__init__(
      field_names=field_names,
      primary_key=primary_key,
      field_types=field_types,
      field_transformers=field_transformers,
      anonymize_fields=anonymize_fields,
      constraints=constraints,
      table_metadata=table_metadata,
      rounding=rounding,
      max_value=max_value,
      min_value=min_value
    )

    self._model_kwargs = {
      'embedding_dim': embedding_dim,
      'generator_dim': generator_dim,
      'discriminator_dim': discriminator_dim,
      'generator_lr': generator_lr,
      'generator_decay': generator_decay,
      'discriminator_lr': discriminator_lr,
      'discriminator_decay': discriminator_decay,
      'batch_size': batch_size,
      'discriminator_steps': discriminator_steps,
      'log_frequency': log_frequency,
      'verbose': verbose,
      'epochs': epochs,
      'pac': pac,
      'cuda': cuda
    }
