import pandas as pd
import os
from sys import *
import xlsxwriter

# application is used to demonstRATes operations on excel file using xlsxwriter

def ExcelCreate(name):
       workbook = xlsxwriter.Workbook(name)
       
       worksheet = workbook.add_worksheet()
       
       worksheet.write('A1','Name')
       worksheet.write('B1','Collage')
       worksheet.write('C1','Mail ID')
       worksheet.write('D1','Mobile')
       
       workbook.close()
       
def main():
       print("----Application Name : "+argv[0])
       
       if(len(argv) != 2):
              print("Error : Invalid number of argumnets")
              exit()
       
       if(argv[1] == "-h") or (argv[1] == "-H"):
              print("This Script used to create excel file and write data into it")
              exit()
              
       if(argv[1] == "-u") or (argv[1] == "-U"):
              print("Usage : AppliactionName Name_Of_File")
              exit()
       try:
              ExcelCreate(argv[1])
       
       except Exception:
              print("Error : Invalid input")
              
if __name__ == "__main__":
       main()
              
       
       
