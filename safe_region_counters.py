from abc import ABC, abstractmethod
import torch
from collections import defaultdict

class SafeRegion(ABC):
    def __init__(self):
        self.hooks = {}
        self.train = True
        self.add = True
        self.count_safe, self.count_notsafe = defaultdict(list), defaultdict(list)
    
    def add_safe(self, key, val):
        if self.add:
            self.count_safe[key].append(val)

    def add_notsafe(self, key, val):
        if self.add:
            self.count_notsafe[key].append(val)
    
    def clear_handles(self):
        for h in self.hooks.values():
            h.remove()
        self.hooks = {}
        # TODO: clear parameters (like min/max)
        
    def clear_counts(self):
        self.count_safe, self.count_notsafe = defaultdict(list), defaultdict(list)

    @abstractmethod
    def get_activation(self, name):
        pass

class LayerMinMax(SafeRegion):
    """ Takes min/max by layer."""
    def __init__(self, generalize_ratio=0.0):
        super().__init__()
        self.min, self.max = {}, {}
        self.generalize_ratio = generalize_ratio

    def get_activation(self, name):
        def count_safe(module, module_in):
            assert isinstance(module_in,tuple) and len(module_in) == 1
            if name not in self.min:
                self.min[name] = float('inf')
            if name not in self.max:
                self.max[name] = float('-inf')

            module_in = module_in[0]
            mi = torch.min(module_in).item()
            ma = torch.max(module_in).item()
            if self.train:
                if mi < self.min[name]:
                    self.min[name] = mi
                if ma > self.max[name]:
                    self.max[name] = ma
            else:
                lower_lim, upper_lim = self.get_bounds(name)
                bounded_check = torch.logical_or(module_in < lower_lim, module_in > upper_lim)
                num_not_safe = torch.sum(bounded_check).item()
                num_all_samples =  torch.numel(bounded_check)
                num_safe = num_all_samples - num_not_safe
                #self.count_safe[name].append(num_safe)#/num_all_samples)
                self.add_safe(name, num_safe)
                self.add_notsafe(name, num_not_safe)
                #self.count_notsafe[name].append(num_not_safe)#/num_all_samples)
        return count_safe

    def get_bounds(self, name):
        if self.min[name] < 0:
            lower_lim = self.min[name] * (1+self.generalize_ratio)
        else:
            lower_lim = self.min[name] * (1-self.generalize_ratio)

        if self.max[name] < 0:
            upper_lim = self.max[name]*(1-self.generalize_ratio)
        else:
            upper_lim = self.max[name]*(1+self.generalize_ratio)

        return lower_lim, upper_lim

class WholeMinMax(SafeRegion):
    """ The most naive method - take min/max of all the layers."""
    def __init__(self, generalize_ratio=0.0):
        super().__init__()
        self.min, self.max = float('inf'), float('-inf')
        self.generalize_ratio = generalize_ratio
        
    def get_activation(self, name):
        def count_safe(module, module_in):
            assert isinstance(module_in,tuple) and len(module_in) == 1
            module_in = module_in[0]
            mi = torch.min(module_in).item()
            ma = torch.max(module_in).item()
            if self.train:
                if mi < self.min:
                    self.min = mi
                if ma > self.max:
                    self.max = ma
            else:
                lower_lim, upper_lim = self.get_bounds()
                bounded_check = torch.logical_or(module_in < lower_lim, module_in > upper_lim)
                num_not_safe = torch.sum(bounded_check).item()
                num_all_samples =  torch.numel(bounded_check)
                num_safe = num_all_samples - num_not_safe
                #self.count_safe[name].append(num_safe)#/num_all_samples)
                #self.count_notsafe[name].append(num_not_safe)#/num_all_samples)
                self.add_safe(name, num_safe)
                self.add_notsafe(name, num_not_safe)
        return count_safe
    
    def get_bounds(self):
        if self.min < 0:
            lower_lim = self.min*(1+self.generalize_ratio)
        else:
            lower_lim = self.min*(1-self.generalize_ratio)
            
        if self.max < 0:
            upper_lim = self.max*(1-self.generalize_ratio)
        else:
            upper_lim = self.max*(1+self.generalize_ratio)
            
        return lower_lim, upper_lim
