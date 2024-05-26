from dataclasses import MISSING, dataclass, field

import hydra
from hydra.core.config_store import ConfigStore

@dataclass
class GeneratorConfig:
    pass

@dataclass
class SchemaGeneratorConfig(GeneratorConfig):
    type: str=MISSING
    generator: str='schema'

@dataclass
class ImageGeneratorConfig(GeneratorConfig):
    generator: str='image'
    dataset: str='mnist'
    n_per_class: int=100
    root_path: str='_data'
    download: bool=True

@dataclass
class Config:
    input_generator: GeneratorConfig = MISSING
    output_generator: GeneratorConfig = MISSING

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group='input_generator', name='schema', node=SchemaGeneratorConfig(type='onehot'))
cs.store(group='input_generator', name='image', node=ImageGeneratorConfig)
cs.store(group='output_generator', name='schema', node=SchemaGeneratorConfig(type='identity'))
cs.store(group='output_generator', name='image', node=ImageGeneratorConfig)

@hydra.main(version_base=None, config_name="config", config_path='conf')
def my_app(cfg: Config) -> None:
    print(cfg)

if __name__ == "__main__":
    my_app()
