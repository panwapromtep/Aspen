# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:19:47 2025

@author: wsangpa1
"""

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
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from CodeLibrary import Simulation

sim = Simulation(AspenFileName= "FlashOperation.bkp", WorkingDirectoryPath= "../FlashOperation", VISIBILITY=True)
print(sim.BLK_COMPR_Get_Discharge_Pressure("COMP1"))

sim.Run()

F3 = sim.STRM_Get_MoleFlow("3")
print(F3)

H3 = sim.STRM_Get_Molar_Enthalpy("3")
print(H3)

'''
print("\n I am going to test if we can get and set the flash drum pressure and temp \n")
flashtemp = sim.BLK_FLASH2_Get_Temperature("FLASH1")
flashpressure = sim.BLK_FLASH2_Get_Pressure("FLASH1")
print("Flash temp is ", flashtemp, " and flash pressure is ", flashpressure)
print("change")
sim.BLK_FLASH2_Set_Temperature("FLASH1", 555)
sim.BLK_FLASH2_Set_Pressure("FLASH1", 200)
print("Flash temp is ", flashtemp, " and flash pressure is ", flashpressure)




print("\n I am going to test if we can get and set the compressor discharge pressure \n")
dischargePres = sim.BLK_COMPR_GET_DISCHARGE_PRES("COMP1")
print("Discharge pressure is ", dischargePres)
print("change")
sim.BLK_COMPR_SET_DISCHARGE_PRES("COMP1", 25)
dischargePres = sim.BLK_COMPR_GET_DISCHARGE_PRES("COMP1")
print("Discharge pressure is ", dischargePres)




print("\n we're gonna save dis shit")
sim.SaveAs("FlashOperation_updated.bkp", True)
'''

print ("Closing Aspen \n \n")

#remember to close Aspen everytime after
#sim.CloseAspen()
