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
                 wdpath: str, #working directory path
                 visiblity: bool = False
                 ):
        self.AspenFile = AspenFile
        self.wdpath = wdpath
        self.visibility = visiblity
    
    def reset(self):
        raise NotImplementedError("The unflatten_params method is not implemented yet.")
        
    def unflatten_params(self, x):
        raise NotImplementedError("The unflatten_params method is not implemented yet.")
    
    def flatten_params(self, x):
        raise NotImplementedError("The flatten_params method is not implemented yet.")
    
    def runSim(self, x):
        raise NotImplementedError("The runSim method is not implemented yet.")
    
    def costFunc(self, x):
        raise NotImplementedError("The costFunc method is not implemented yet.")
        
    def run_obj(self, x):
        res = self.runSim(x)
        return self.costFunc(res)