import os
import json
import numpy as np
import pandas as pd
from time import time

def _inames(prefix):
    # rando40 => rando40_dat.csv, rando40_opt.csv, rando40_inf.json
    r = [prefix+"_"+x for x in "dat.csv opt.csv opt2.csv inf.json".split()]
    return tuple(r)

class MyDataSet():

    def __init__(self, prefix=None, filename=None,dir=None, X=None):
        if prefix is not None:
            self.restore(prefix=prefix,dir=dir)
        elif filename is not None:
            self.read_csv(filename=filename,dir=dir)
        if isinstance(X,np.ndarray):
            self.data=X
            self.description="numpy array"
            self.opt_sse=None
            self.opt2_sse=None
            self._called_as="(no info)"

    def read_csv(self, filename, dir=None):
        if dir is None:
            dir = "."   
        csvfile = os.path.join(dir,filename) 
        assert os.path.isfile(csvfile), f"file not found: {csvfile}"
        df = pd.read_csv(csvfile, header=None)
        self.data=df.to_numpy()        
        self.description="no details available"
        self.opt_sse=None
        self.opt2_sse=None
        self._called_as="(no info)"

    def restore(self, prefix=None, dir=None):
        """load data set from files
        """

        if dir is None:
            dir = "."
        datfile,optfile,opt2file,inffile=[os.path.join(dir,x) for x in _inames(prefix)]

        assert os.path.isfile(datfile), f"file not found: {datfile}"
        assert os.path.isfile(inffile), f"file not found: {inffile}"

        # load parameter files
        if os.path.isfile(inffile):
            with open(inffile) as json_file:
                data = json.load(json_file)
                #print(data)
                if (data["opt_known"]):
                    assert os.path.isfile(optfile), f"file not found: {optfile}"

                self.description=data["description"]
                self.opt_sse=data["opt_SSE"]
                self.opt2_sse=data["opt2_SSE"]
                self._called_as=data["called_as"]
        else:
                self.description="no details available"
                self.opt_sse=None
                self.opt2_sse=None
                self._called_as=data["called_as"]            

        df = pd.read_csv(datfile, header=None)
        self.data=df.to_numpy()

        if (data["opt_known"]):
            df = pd.read_csv(optfile, header=None)
            self.optimum = df.to_numpy()
        else:
            self.optimum = None

        if (data["opt2_known"]):
            df = pd.read_csv(opt2file, header=None)
            self.optimum2 = df.to_numpy()
        else:
            self.optimum2 = None        


    def store(self,  prefix=None, dir=None):
        """store data set in three files:

        * ..._dat.csv
        * ..._opt.csv
        * ..._inf.json
        """
        
        # obscure but unique prefix: time() in base=36 notation 
        def codi():
            t= round(time())
            a=np.base_repr(t,base=36)
            return a
        if dir is None:
            dir = "."
        if prefix is None:
            prefix = codi()
                
        # filenames
        datfile,optfile,opt2file,inffile=[os.path.join(dir,x) for x in _inames(prefix)]
        
        # store parameter file
        pars=self.get_params()
        with open(inffile, 'w', encoding='utf-8') as f:
            json.dump(pars, f, ensure_ascii=False, indent=4)
        
        # store data file
        data = self.get_data()
        data = pd.DataFrame(data)
        data.to_csv(datfile, header=False, index=False)

        # store optimum (or good) solution file
        if self.get_opt_known():
            data = self.get_optimum()
            data = pd.DataFrame(data)
            data.to_csv(optfile, header=False, index=False)

        # store optimum2
        if self.get_opt2_known():
            data = self.get_optimum2()
            data = pd.DataFrame(data)
            data.to_csv(opt2file, header=False, index=False)

    def get_opt_known(self):
        """do we know the optimum for one k-value?"""
        return not self.optimum is None

    def get_opt2_known(self):
        """do we know the optimum for k=n/2?"""
        return not self.optimum2 is None
    def get_data(self):
        """return data set"""
        return self.data.astype(np.float64)
    def get_d(self):
        """return number of features (d)"""
        return self.get_data().shape[1]

    def get_n(self):
        """return number of samples"""
        return self.get_data().shape[0]

    def get_opt_k(self):
        """get the k for which the optimum is known"""
        if self.optimum is None:
            raise Exception("no optimal k known!")
        else:
            return self.optimum.shape[0]

    def get_opt2_k(self):
        """get the k for which the optimum is known"""
        if self.optimum2 is None:
            raise Exception("no optimal2 k known!")
        else:
            return self.optimum2.shape[0]
    def get_opt2_sse(self):
        """SSE for case that k = n/2 and all data pairs have same distance"""
        assert self.get_opt2_known(), "Error for k=n/2 problem is not known. "+self.no_sse2_reason
        return self.opt2_sse

    def get_opt_sse(self):
        """get the SSE of the optimal solution
        precondition: Optimal solution is known
        """
        assert self.get_opt_known(), "no optimal SSE known!"
        return self.opt_sse

    def get_opt_known(self):
        """do we know the optimum for one k-value?"""
        return not self.optimum is None

    def get_opt2_known(self):
        """do we know the optimum for k=n/2?"""
        return not self.optimum2 is None

    def get_optimum(self):
        """get the optimal solution
        precondition: Optimal solution is known
        """
        assert self.get_opt_known(), "no optimum known!"
        return self.optimum

    def get_optimum2(self):
        """get the optimal solution for k=n/2 case
        precondition: Optimal solution2 is known
        """
        assert self.get_opt2_known(), "no optimum for k=n/2 problem known!"
        return self.optimum2

if __name__ == "__main__":
    D = MyDataSet()
    D.restore(prefix="nogo",dir="/tmp")
    print(D.get_data())