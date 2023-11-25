import pandas as pd

# application is used to demonstRATes operations on excel file using library

excel_file = 'Marvellous.xlsx'
batches = pd.read_excel(excel_file)

print(batches.head())

batches_sheet1 = pd.read_excel(excel_file, sheet_name=0, index_col=0)
print(batches_sheet1.head())

xlsx = pd.ExcelFile(excel_file)
batches_sheet = []

for sheet in xlsx.sheet_names:
       print(sheet)
       batches_sheet.append(xlsx.parse(sheet))
       
batches = pd.concat(batches_sheet)
print(batches)

