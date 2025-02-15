# Gradient Descent Module
import pickle
import numpy as np
from CodeLibrary import Simulation

class AspenSim():
    """
    A class to represent an Aspen simulation.
    Attributes
    ----------
    AspenFile : str
        The path to the Aspen file.
    runSim : function
        A function to run the simulation.
    costFunc : function
        A function to calculate the cost based on the simulation results.
    wdpath : str
        The path to the working directory where you want to do your work
    Methods
    -------
    run_obj(x):
        Runs the simulation with the given parameters and returns the cost.
    """
    def __init__(self, 
                 AspenFile: str, 
                 runSim, 
                 costFunc,
                 wdpath: str #working directory path
                 ):
        self.AspenFile = AspenFile
        self.runSim = runSim
        self.costFunc = costFunc
        self.wdpath = wdpath
        
    def run_obj(self, x):
        self.runSim(self.AspenFile, self.wdpath, x)
        return self.costFunc(self.AspenFile)