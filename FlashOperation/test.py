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


sim = Simulation(AspenFileName= "FlashOperation.bkp", WorkingDirectoryPath= "../FlashOperation", VISIBILITY=False)


print("\n I am going to test if we can get and set the flash drum pressure and temp \n")
flashtemp = sim.BLK_FLASH2_Get_Temperature("FLASH1")
flashpressure = sim.BLK_FLASH2_Get_Pressure("FLASH1")
print("Flash temp is ", flashtemp, " and flash pressure is ", flashpressure)
print("change")
sim.BLK_FLASH2_Set_Temperature(555)
sim.BLK_FLASH2_Set_Pressure(8008)
print("Flash temp is ", flashtemp, " and flash pressure is ", flashpressure)



print("\n I am going to test if we can get and set the compressor discharge pressure \n")
dischargePres = sim.BLK_COMPR_GET_DISCHARGE_PRES("COMP1")
sim.print_dictionary(dischargePres)
print("change")
sim.BLK_COMPR_SET_DISCHARGE_PRES("COMP1", 25)
sim.print_dictionary(sim.BLK_COMPR_GET_DISCHARGE_PRES("COMP1"))




print("\n we're gonna save dis shit")
sim.SaveAs("FlashOperation_updated.bkp", True)

print ("Closing Aspen \n \n")

#remember to close Aspen everytime after
sim.CloseAspen()
