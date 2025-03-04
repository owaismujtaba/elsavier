
from src.dataset.data_reader import DatasReader
from src.dataset.data_loader import DataLoader

filepath = '/home/owaismujtaba/projects/elsavier/Data/F10.edf'

data_reader = DatasReader(filepath)
data_loader = DataLoader(data_reader)