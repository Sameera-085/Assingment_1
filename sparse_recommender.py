import numpy as np
import scipy.sparse as sp
import pytest

class SparseMatrix:
    def __init__(self, row, col):#Initializing the Sparse Matrix
        self.matrix = sp.lil_matrix((row, col))

    def set_value(self, row, col, v1):# V1 is the value we are setting at (row,col)
        self.matrix[row, col] = v1

    def get_value(self, row, col):# Function to get the value at (row,col)
        return self.matrix[row, col]

    def recommend(self, Multiplication_Vector):# Multiplying the vector with sparse matrix to get recommendations
        Vector_Result = self.matrix.dot(Multiplication_Vector)
        return Vector_Result

    def add_(self, SparseMatrix_2):# Function for Addition of Sparse Matrix with another Sparse Matrix (SparseMatrix_2)
        
            if self.matrix.shape != SparseMatrix_2.shape:
                print("Error: Shapes of Matrices didn't match")
                return None

            Matrix_1=self.matrix + SparseMatrix_2
            return Matrix_1
        

# Dense Matrix: In dense matrix zero and Non-zero elements are stored, where as in sparse matrix only Non-zero elements are stored. 
    def to_dense(self): # Function to convert a Sparse Matrix into Dense Matrix
        dense_matrix = [[0 for i in range(self.matrix.shape[1])] for i in range(self.matrix.shape[0])]
        for row in range(self.matrix.shape[0]):
            for col in range(self.matrix.shape[1]):
                dense_matrix[row][col] = self.get_value(row, col)

        return dense_matrix



# Additional Test Cases-> Additional Operations on Sparse Matrix


# Printing Non Zero elements in Sparse Matrix with their position 
    def non_zero_Positions(self):
        for row in range(self.matrix.shape[0]):
            for col in range(self.matrix.shape[1]):
                value = self.get_value(row, col)
                if value != 0:
                    print(f"Value at ({row}, {col}): {value}")

# Printing all Non Zero elements in Sparse Matrix as a list with their Sum.  
  
    def List_Non_Zero(self):
        l1=[]
        for row in range(self.matrix.shape[0]):
            for col in range(self.matrix.shape[1]):
                value = self.get_value(row, col)
                if value != 0:
                   l1.append(value)
        print(l1)
        return sum(l1)
    

# transposing Sparse matrix(rows to columns and columns to rows)
    def Transpose_sparse(self):
        trans=self.matrix.transpose()
        return trans

# Minimum No of elements to change into Non-zero so that Sparse Matrix has more non-zero elements.
    def Convert_Sparse(self):
        s=0
        ns=0
        c=0
        for row in range(self.matrix.shape[0]):
            for col in range(self.matrix.shape[1]):
                value = self.get_value(row, col)
                if value != 0:
                   ns=ns+1
                else:
                   s=s+1
        while s>ns:
            s=s-1
            ns=ns+1
            c=c+1
        return c

# To mutliply a matrix with a Sparse Matrix, While checking the condition that No of columns in matrix should match with no of rows of Sparse matrix.
    def To_Multiply(self,Matrix_For_Multiply):
        #result_1=SparseMatrix(3,3)
        if self.matrix.shape != Matrix_For_Multiply.shape:
                print("Error: Length of columns of Matrix1 and rows of Matrix didn't match")
        else:
            result_1=Matrix_For_Multiply @ self.matrix.tocsr()
        
            return result_1

