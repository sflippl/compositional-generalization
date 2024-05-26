from dataclasses import MISSING, dataclass, field
from typing import Union, List, Any

import hydra
from hydra.core.config_store import ConfigStore
import numpy as np

from simple_compositional_tasks.tasks.get_tasks import get_tasks
from simple_compositional_tasks.embeddings.get_embedding import get_embeddings
from simple_compositional_tasks.analysis.get_analysis import get_analysis
from simple_compositional_tasks.models.get_module import get_module

def create_dataset(task, embedding):
    x = embedding[0](task.x)
    y = embedding[1](task.y)
    if 'val_split' not in task.df.columns:
        val_split = np.where(task.df['train'], 'train', 'test') # This will enable crosslabeling from task and embedding later
    else:
        val_split = task.df['val_split']
    val_split_values = np.unique(val_split)
    val_split_index = (val_split.reshape(-1,1)==val_split_values.reshape(1,-1)).nonzero()[1]
    return (x[task.df['train']], y[task.df['train']]), (x, y, val_split_index), val_split_values

@dataclass
class GeneratorConfig:
    generator: str=MISSING
    permutation: str='none'
    permutation_seed: int|None=None
    permutation_index: int|None=None

@dataclass(kw_only=True)
class SchemaGeneratorConfig(GeneratorConfig):
    generator: str='schema'
    type: str=MISSING

@dataclass
class ImageGeneratorConfig(GeneratorConfig):
    generator: str='image'
    dataset: str='mnist'
    n_training_imgs: int=100
    n_training_samples: int=50
    n_training_samples_total: int|None=None
    n_validation_samples_1: int=100
    n_validation_samples_2: int=100
    seed: int|None=None
    root: str='_data'
    download: bool=True
    distance: int=28
    separate_channels: bool=False
    augmentation: list[str]=field(default_factory=lambda: [])

@dataclass
class TaskConfig:
    task: str=MISSING

@dataclass
class AdditionTaskConfig(TaskConfig):
    task: str='addition'
    training_selection: str='comps'
    task_values: list[list[int]]=field(default_factory=lambda: [[-2,-1,0,1,2], [-2,-1,0,1,2]])
    training_comps: list[list[int]]=field(default_factory=lambda: [[2], [2]])
    exceptions: list[list]=field(default_factory=lambda: [])
    n_training_instances: int=2
    instances_seed: int|None=None

@dataclass
class TransitiveOrderingTaskConfig(TaskConfig):
    task: str='transitive_ordering'
    n_items: int=7
    train_sds: list[int]=field(default_factory=lambda: [1])
    exceptions: list[list]=field(default_factory=lambda: [])

@dataclass(kw_only=True)
class ContextDependenceTaskConfig(TaskConfig):
    task: str='context_dependence'
    contexts: int=MISSING
    features: list[int]=MISSING
    leftout_conjunctions: list[dict[int, list[int]]]=MISSING
    context_to_feature: list[int]=MISSING
    feature_to_response: list[list[int]]=MISSING

@dataclass(kw_only=True)
class CDStimulusCompositionTaskConfig(ContextDependenceTaskConfig):
    contexts: int=2
    features: list[int]=field(default_factory=lambda: [2,2])
    leftout_conjunctions: list[dict[int, list[int]]]=field(default_factory=lambda: [{1: [0], 2: [1]}])
    context_to_feature: list[int]=field(default_factory=lambda: [0,1])
    feature_to_response: list[list[int]]=field(default_factory=lambda: [[-1,1],[-1,1]])

@dataclass(kw_only=True)
class CDStimulusComposition2TaskConfig(ContextDependenceTaskConfig):
    contexts: int=4
    features: list[int]=field(default_factory=lambda: [4,4])
    leftout_conjunctions: list[dict[int, list[int]]]=field(default_factory=lambda: [{1: [0,1], 2: [2,3]}])
    context_to_feature: list[int]=field(default_factory=lambda: [0,0,1,1])
    feature_to_response: list[list[int]]=field(default_factory=lambda: [[-1,-1,1,1],[-1,-1,1,1]])

@dataclass(kw_only=True)
class CDContextSubstitutionTaskConfig(ContextDependenceTaskConfig):
    contexts: int=4
    features: list[int]=field(default_factory=lambda: [6,6])
    leftout_conjunctions: list[dict[int, list[int]]]=field(default_factory=lambda: [{0: [0,2], 1: [0,3]}, {0: [0,2], 2: [0,3]}])
    context_to_feature: list[int]=field(default_factory=lambda: [0,0,1,1])
    feature_to_response: list[list[int]]=field(default_factory=lambda: [[-1,-1,-1,1,1,1],[-1,-1,-1,1,1,1]])

@dataclass(kw_only=True)
class TransitiveEquivalenceTaskConfig(TaskConfig):
    task: str='transitive_equivalence'
    classes: list[int]=field(default_factory=lambda: [2, 2])
    train_items: list[int]=field(default_factory=lambda: [1, 1])

@dataclass
class ModelConfig:
    model_type: str=MISSING
    outp_dim: str|int = 'automatic'

@dataclass
class NetworkConfig:
    network_type: str=MISSING

@dataclass
class LightningTrainerConfig:
    accelerator: str='cpu'
    devices: int=1
    training_seed: int|None=None
    batch_training: bool=False
    batch_testing: bool=False
    train_batch_size: int=128
    test_batch_size: int=128
    lr: float=0.1
    momentum: float=0.0
    criterion: str='mse'
    max_epochs: int=500
    log_every_n_steps: int=1
    check_val_every_n_epoch: float|int=1
    analyse_every_n_epoch: int=1
    analyse_first_n_epochs: int|None=None
    analyse_on_epochs: list[int]|None=None
    auto_lr_find: bool|str=False
    auto_lr_find_num_steps: int=100
    early_stopping: bool=False
    early_stopping_stopping_threshold: float|None=1e-8
    early_stopping_divergence_threshold: float|None=20
    num_workers: int=0
    inference_mode: bool=True
    optimizer: str='sgd'
    scheduler: str='none'

@dataclass
class DenseNetConfig(NetworkConfig):
    network_type: str='densenet'
    hdims: list[int]=field(default_factory=lambda: [100])
    bias: bool=True
    log_scaling: float=0.
    linear_readout: bool=False
    seed: int|None=None

@dataclass
class ConvNetConfig(NetworkConfig):
    network_type: str='convnet'
    conv_layers: list[int]=field(default_factory=lambda: [32, 32, 64, 64])
    dense_layers: list[int]=field(default_factory=lambda: [512, 1024])
    in_channels: int=1
    kernel_size: int=5
    pool_size: int=2
    seed: int|None=None

@dataclass
class ResNetConfig(NetworkConfig):
    network_type: str='resnet'
    seed: int|None=None
    conv_channels: list[int]=field(default_factory=lambda:[64, 128, 256, 512])

@dataclass
class ViTConfig(NetworkConfig):
    network_type: str='vit'
    seed: int|None=None
    patch_size: tuple[int]=field(default_factory=lambda: (8,8))
    heads: int=8
    mlp_dim: int=512
    dim: int=512
    depth: int=6

@dataclass(kw_only=True)
class LightningConfig(ModelConfig):
    defaults: List[Any] = field(default_factory=lambda: [{'network': 'densenet'}])
    network: NetworkConfig = MISSING
    trainer_config: LightningTrainerConfig = field(default_factory=lambda: LightningTrainerConfig())
    model_type: str='lightning'

@dataclass(kw_only=True)
class KernelMachineConfig(ModelConfig):
    model_type: str='kernel_machine'
    sims: dict[str, float|str]=field(default_factory=lambda: {'0': 0., '1': 'np.linspace(0., 0.5, 51)', '2': 1.})
    C: float|str=1e10
    objective: str='regression'
    equivariant: bool=True
    kernel_file: str|None=None

@dataclass(kw_only=True)
class NTKConfig(ModelConfig):
    defaults: List[Any] = field(default_factory=lambda: [{'network': 'densenet'}])
    model_type: str='ntk'
    network: NetworkConfig = MISSING
    C: float|str=1e10
    objective: str='regression'

@dataclass
class Config:
    input_generator: GeneratorConfig = MISSING
    output_generator: GeneratorConfig = MISSING
    task: TaskConfig = MISSING
    model: ModelConfig = MISSING
    metrics: list[str] = field(default_factory=lambda: [])
    analysis: list[str] = field(default_factory=lambda: ['additivity'])

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group='task', name='addition', node=AdditionTaskConfig)
cs.store(group='task', name='transitive_ordering', node=TransitiveOrderingTaskConfig)
cs.store(group='task', name='context_dependence', node=ContextDependenceTaskConfig)
cs.store(group='task', name='cd_stimulus_composition', node=CDStimulusCompositionTaskConfig)
cs.store(group='task', name='cd_stimulus_composition_2', node=CDStimulusComposition2TaskConfig)
cs.store(group='task', name='cd_context_substitution', node=CDContextSubstitutionTaskConfig)
cs.store(group='task', name='transitive_equivalence', node=TransitiveEquivalenceTaskConfig)
cs.store(group='embedding.input_generator', name='schema', node=SchemaGeneratorConfig(type='one_hot'))
cs.store(group='embedding.input_generator', name='image', node=ImageGeneratorConfig)
cs.store(group='embedding.output_generator', name='schema', node=SchemaGeneratorConfig(type='identity'))
cs.store(group='embedding.output_generator', name='image', node=ImageGeneratorConfig)
cs.store(group='model', name='lightning', node=LightningConfig)
cs.store(group='model/network', name='densenet', node=DenseNetConfig)
cs.store(group='model/network', name='convnet', node=ConvNetConfig)
cs.store(group='model/network', name='resnet', node=ResNetConfig)
cs.store(group='model/network', name='vit', node=ViTConfig)
cs.store(group='model.trainer_config', name='lightning', node=LightningTrainerConfig)
cs.store(group='model', name='kernel_machine', node=KernelMachineConfig)

@hydra.main(version_base=None, config_name="config", config_path='conf')
def my_app(cfg: Config) -> None:
    task = get_tasks(cfg.task)
    embedding = get_embeddings(cfg.embedding, task)
    analysis = get_analysis(task, cfg.analysis)
    train_data = embedding.create_dataset('train')
    val_data = embedding.create_dataset('val')
    val_splits = embedding.val_splits
    if train_data[0][1].ndim==0:
        outp_dim=1
        flatten=True
    else:
        outp_dim=train_data[0][1].shape[-1]
        flatten=False
    model = get_module(
        cfg.model, analysis, metrics=cfg.metrics,
        inp_dim=train_data[0][0].shape[-1], outp_dim=outp_dim,
        flatten=flatten, val_splits=val_splits, test_input=train_data[0][0],
        comps=embedding.input_embedding.comps
    )
    model.fit(train_data, val_data)


if __name__ == "__main__":
    my_app()
