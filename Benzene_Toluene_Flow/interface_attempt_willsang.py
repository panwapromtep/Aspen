# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 23:16:30 2025

@author: wsangpa1
"""

#from ast import AugAssign
import sys
import os


print("hi, this is will's first attempt at getting the inteface to work")
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)



from CodeLibrary import Simulation

print("Opening ASPEN file \n \n")

sim = Simulation(AspenFileName= "Flash_Toluene_Benzene.bkp", WorkingDirectoryPath= "../Benzene_Toluene_Flow" ,VISIBILITY=True)

print("Aspen is now open \n \n")

#HEATEROutletTemperature = sim.BLK_HEATER_Get_OutletTemperature("B2")
#print("The outlet temp is ", HEATEROutletTemperature, "\n \n")

HEATERInputDictionary = sim.BLK_HEATER_GET_ME_ALL_INPUTS_BACK("B2")
sim.print_dictionary(HEATERInputDictionary)


print("We are going to change the temperature of the heater \n \n")
sim.BLK_HEATER_Set_Temperature("B2", 200)

print("it should've changed now")
HEATERInputDictionary = sim.BLK_HEATER_GET_ME_ALL_INPUTS_BACK("B2")
sim.print_dictionary(HEATERInputDictionary)


print("we're gonna save dis shit")
sim.SaveAs("test.bkp", True)


print ("Closing Aspen \n \n")

#remember to close Aspen everytime after
#sim.CloseAspen()