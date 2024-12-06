# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 12:52:46 2024

@author: Leon Schöne
"""

import numpy as np
import pandas as pd

def GenerateLatexTable(inputArray, columnHeadings, formatStrings, fileName):
    
    # Determine the number of rows and columns
    numRows, numCols = inputArray.shape
    
    # Determine the number of headings. Note that this can be different
    # to the number of columns if \multicolumn is used.
    numHeadings = len(columnHeadings)
    
    # Generate the start of the table and column alignment specifiers.
    tableString = "\\begin{tabular}{|"
    for i in range(0, numCols):
        tableString += "c|"
        
    tableString += "}\n\hline\n"
    
    # Generate the table headings
    # Note that we handle the last heading differently.
    for i in range(0, numHeadings):
        currentHeading = columnHeadings[i]
        tableString += currentHeading
        if (i < numHeadings-1):
            tableString += " & "
        else:
            tableString += "\\\\ \n"
            
    # Add a horizontal line below the column headings.
    tableString += "\hline \n"
    
    # Generate the table body.
    # Note that for each row, we handle the last column differently.
    for row in range(0, numRows):
        for col in range(0, numCols):
            currentString = f"{inputArray[row, col]:{formatStrings[col]}}"
            if (col < numCols-1):
                currentString += " & "
            else:
                currentString += "\\\\ \n"
            
            tableString += currentString
    
    # And remember to end the "tabular" environment.
    tableString += "\\hline \n\\end{tabular}\n"
    
    # Print the final string to the console so the user can copy and paste 
    # it into their Latex file
    # print(tableString)
    with open(f'../output_tables/{fileName}.txt', 'w') as file:
        file.write(tableString)
        
        

def GenerateLatexTable2(inputData, columnHeadings, formatStrings, fileName):
    # Check if the input is a DataFrame
    if isinstance(inputData, pd.DataFrame):
        # Assume the first column contains strings; remove it and store separately
        string_column = inputData.iloc[:, 0].to_numpy()  # First column as strings
        numeric_data = inputData.iloc[:, 1:].to_numpy()  # Remaining numeric data
        
        # Use the cleaned numeric data for further processing
        inputArray = numeric_data
    elif isinstance(inputData, np.ndarray):
        inputArray = inputData
        string_column = None
    else:
        raise ValueError("Input data must be a Pandas DataFrame or a NumPy Array.")
    
    # Determine the number of rows and columns
    numRows, numCols = inputArray.shape
    
    # Determine the number of headings. Note that this can be different
    # to the number of columns if \multicolumn is used.
    numHeadings = len(columnHeadings)
    
    # Generate the start of the table and column alignment specifiers.
    tableString = "\\begin{tabular}{|"
    for i in range(0, numCols):
        tableString += "c|"
        
    tableString += "}\n\hline\n"
    
    # Generate the table headings
    # Note that we handle the last heading differently.
    for i in range(0, numHeadings):
        currentHeading = columnHeadings[i]
        tableString += currentHeading
        if (i < numHeadings-1):
            tableString += " & "
        else:
            tableString += "\\\\ \n"
            
    # Add a horizontal line below the column headings.
    tableString += "\hline \n"
    
    # Generate the table body.
    # Note that for each row, we handle the last column differently.
    for row in range(0, numRows):
        row_strings = []
        
        # If there's a string column, add the string at the beginning of the row
        if string_column is not None:
            row_strings.append(string_column[row])
        
        for col in range(0, numCols):
            currentString = f"{inputArray[row, col]:{formatStrings[col]}}"
            row_strings.append(currentString)
        
        # Join the row's columns with '&' and add the ending '\\'
        tableString += " & ".join(row_strings) + " \\\\ \n"
    
    # And remember to end the "tabular" environment.
    tableString += "\\hline \n\\end{tabular}\n"
    
    # Print the final string to the console so the user can copy and paste 
    # it into their Latex file
    # print(tableString)
    with open(f'../output_tables/{fileName}.txt', 'w') as file:
        file.write(tableString)
        
        
def GenerateLatexTable3(inputArray, columnHeadings, fileName):
    
    # Determine the number of rows and columns
    numRows, numCols = inputArray.shape
    
    # Determine the number of headings. Note that this can be different
    # to the number of columns if \multicolumn is used.
    numHeadings = len(columnHeadings)
    
    # Generate the start of the table and column alignment specifiers.
    tableString = "\\begin{tabular}{|"
    for i in range(0, numCols):
        tableString += "c|"
        
    tableString += "}\n\hline\n"
    
    # Generate the table headings
    # Note that we handle the last heading differently.
    for i in range(0, numHeadings):
        currentHeading = columnHeadings[i]
        tableString += currentHeading
        if (i < numHeadings-1):
            tableString += " & "
        else:
            tableString += "\\\\ \n"
            
    # Add a horizontal line below the column headings.
    tableString += "\hline \n"
    
    # Generate the table body.
    # Note that for each row, we handle the last column differently.
    for row in range(0, numRows):
        for col in range(0, numCols):
            if isinstance(inputArray[row, col], str):
                currentString = inputArray[row, col]
            else:
                currentString = f"{inputArray[row, col]:.3f}"
                
            if (col < numCols-1):
                currentString += " & "
            else:
                currentString += "\\\\ \n"
            
            tableString += currentString
    
    # And remember to end the "tabular" environment.
    tableString += "\\hline \n\\end{tabular}\n"
    
    # Print the final string to the console so the user can copy and paste 
    # it into their Latex file
    # print(tableString)
    with open(f'../output_tables/{fileName}.txt', 'w') as file:
        file.write(tableString)