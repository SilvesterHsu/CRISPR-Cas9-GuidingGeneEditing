import pandas as pd
import numpy as np


class LocalLab:
    def __init__(self, file):
        self.file = file
        
        self.data = self._load_data()
        self._drop_nan('qPCR')
        self._drop_nan('dom_tss_pos')
        self._drop_nan('GUIDE_START')
        self._filter_out_guide()
        
        self._transform_TSS()
        self._transform_location()
        self._transform_chr()
        self._transform_select()

    def _drop_nan(self, target_column='qPCR'):
        self.data = self.data[~np.isnan(self.data[target_column])]

    def _load_data(self, sep='	'):
        return pd.read_table(self.file, sep=sep)

    def _filter_out_guide(self, guide_length=23):
        self.data = self.data[self.data['GUIDE'].map(lambda x: len(x) == guide_length)]
    
    def _transform_TSS(self):
        self.data['TSS'] = self.data['dom_tss_pos'].map(lambda x: int(x))
        
    def _transform_location(self):
        self.data['LOCATION'] = self.data['GUIDE_START'].map(lambda x: int(x))
        
    def _transform_chr(self):
        self.data['CHR'] = self.data['CHR'].map(lambda s: int(s[3:]))
        
    def _transform_select(self):
        self.data = self.data[['GUIDE','qPCR','CONTROL','CHR','STRAND','TSS','LOCATION']]

class LocalChopchop:
    def __init__(self,file):
        self.file = file
        self.data = self._load_data()
        self._filter_out_guide()
        
        self._transform_TSS()
        self._transform_location()
        self._transform_chr()
        self._transform_strand()
        self._transform_select()
        
    def _load_data(self):
        return pd.read_csv(self.file).rename(columns={'Target sequence':'GUIDE'})
    
    def _filter_out_guide(self, guide_length=23):
        self.data = self.data[self.data['GUIDE'].map(lambda x: len(x) == guide_length)]
    
    def _transform_TSS(self):
        self.data['TSS'] = self.data['Batch Zone'].map(lambda s: s.split(':')[1].split('-'))
        self.data['TSS'] = self.data['TSS'].map(lambda x: int((int(x[0])+int(x[1]))/2))
        
    def _transform_location(self):
        self.data['LOCATION'] = self.data['Genomic location'].map(lambda s: int(s.split(':')[1]))
        
    def _transform_chr(self):
        self.data['CHR'] = self.data['Batch Zone'].map(lambda s: int(s.split(':')[0][3:]))
        
    def _transform_strand(self):
        self.data['STRAND'] = self.data['Strand'].map(lambda s: 1 if s == '+' else -1)
        
    def _transform_select(self):
        self.data = self.data[['GUIDE','Efficiency','CHR','STRAND','TSS','LOCATION']].rename(columns={'Efficiency': 'EFFICIENCY'})
        