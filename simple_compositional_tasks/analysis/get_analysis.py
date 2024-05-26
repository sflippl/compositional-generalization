from .additivity_analysis import AdditivityAnalysis
from .similarity_analysis import SimilarityAnalysis
from .analysis_formats import CSVDataFrame

class Analysis:
    def __init__(self, task, cfg):
        self.analyses = []
        if 'additivity' in cfg:
            self.analyses.append(AdditivityAnalysis(task.x, task.df))
        if 'feature_sim' in cfg:
            self.analyses.append(SimilarityAnalysis(type='features'))
        if 'ntk_sim' in cfg:
            self.analyses.append(SimilarityAnalysis(type='ntk'))
        if 'output' in cfg:
            self.analyses.append(OutputAnalysis)

    def validation_step(self, model, x, y, yhat, row):
        for analysis in self.analyses:
            analysis.validation_step(model, x, y, yhat, row)
    
    def on_validation_epoch_end(self):
        objs = []
        for analysis in self.analyses:
            new_objs = analysis.on_validation_epoch_end()
            objs.extend(new_objs)
        return objs

    def __call__(self, x, preds, splits, y=None):
        objs = []
        for analysis in self.analyses:
            assert isinstance(analysis, (AdditivityAnalysis,)) # for now we can only call directly for additivity analysis
            new_objs = analysis(x, preds, splits, y)
            objs.extend(new_objs)
        return objs

def get_analysis(task, cfg):
    return Analysis(task, cfg)

class OutputAnalysis:
    def __init__(self):
        super().__init__()
    
    def __call__(self, x, preds, splits, y):
        return [CSVDataFrame({
            'x': list(x),
            'preds': list(preds),
            'splits': list(splits),
            'y': y
        }, file_name='output')]
