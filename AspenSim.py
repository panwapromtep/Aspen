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
                 runSim: function,
                 costFunc: function,
                 wdpath: str #working directory path
                 ):
        self.AspenFile = AspenFile
        self.runSim = runSim
        self.costFunc = costFunc
        self.wdpath = wdpath
        
    def run_obj(self, x):
        self.runSim(self.AspenFile, self.wdpath, x)
        return self.costFunc(self.AspenFile)
    

def runRefrig(AspenFile, wdpath, x):
    """
    Runs the refrigeration simulation with the given parameters.

    Args:
        AspenFile (str): aspen file name
        wdpath (str): path to the aspen file
        
        x (dictionary): Categorizes different types of process blocks (e.g., FLASH2, HEATER) 
        and maps them to their names with an associated parameter list
        
        Structure:
            x = {
                "BLOCK_TYPE": {"BLOCK_NAME": [PARAM1, PARAM2]}
                "FLASH2": {"BLOCK_NAME": [temp, pressure]}
                "HEATER": {"BLOCK_NAME": [temp, pressure]}
                "COMPR": {"BLOCK_NAME": [discharge_pressure]}
            }
    Returns:
        obj: The result of the simulation.
    """
    
    #todo check to see if WorkingDirectoryPath works as intended
    sim = Simulation(AspenFileName= AspenFile, WorkingDirectoryPath= wdpath , VISIBILITY=False)
    
    #loop through flashdrums and set values
    for blockname in x["FLASH2"]:
        sim.BLK_FLASH2_Set_Temperature(blockname, x["FLASH2"][blockname][0])
        sim.BLK_FLASH2_Set_Pressure(blockname, x["FLASH2"][blockname][1])
        
    #loop thorugh heaters and set values
    for blockname in x["HEATER"]:
        sim.BLK_HEATER_Set_Temperature(blockname, x["HEATER"][blockname][0])
        sim.BLK_HEATER_Set_Pressure(blockname, x["HEATER"][blockname][1])
        
    #loop thorugh compressors and set values
    for blockname in x["COMPR"]:
        sim.BLK_HEATER_Set_Discharge_Pressure(blockname, x["COMPR"][blockname][0])
        
        
    #run the simulation
    sim.DialogSuppression(TrueOrFalse= True)
    sim.Run()

    
    F3 = sim.STRM_Get_MoleFlow("3")
    F5 = sim.STRM_Get_MoleFlow("5")
    F7 = sim.STRM_Get_MoleFlow("7")
    
    H3 = sim.STRM_Get_Molar_Enthalpy("3")
    
    
    # Add your simulation code here
    res = None  # Replace with actual simulation result
    return res

def costFunc_refrig(res: np.ndarray):
    """
    Calculates the cost based on the simulation results.

    Args:
        AspenFile (str): The path to the Aspen file.

    Returns:
        float: The cost of the simulation.
    """
    # Add your cost calculation code here
    cost = None  # Replace with actual cost calculation
    return cost


'''
class FlashDrum():
    """
    A class to represent an Aspen Flash Drum.
    Attributes (limited to what we want to change right now, I can expand this later)
    ---------
    blockname : str
        name of the block in aspen
    temp : float
        the temperature
    pressure : float
        the temperature
        
    Methods
    ------
    set_temp(sim):
        sets the temperature
    set_pressure(sim):
        sets the pressure
    """
    
    def __inti__ (self, 
                  blockname: str,
                  temp: float,
                  pressure: float):
        self.name = blockname
        self.temp = temp
        self.pressure = pressure
        
    def set_temp(self, sim: Simulation, temp: float):
        sim.BLK_FLASH2_Set_Temperature(self.name, temp)
        
    def set_pressure(self, sim: Simulation, pressure: float):
        sim.BLK_FLASH2_Set_Pressure(self.name, pressure)
    
        
'''