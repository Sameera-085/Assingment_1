# Create the SparseMatrix instance

from sparse_recommender import SparseMatrix
import numpy as np
import scipy.sparse as sp
import pytest

sparse_matrix = SparseMatrix(3, 3)

x=SparseMatrix(3,3)
sparse_matrix.set_value(0, 1, 1)
sparse_matrix.set_value(1, 2, 2)
sparse_matrix.set_value(2, 2, 3)


recommendations_vector = np.array([2,4,6])

SparseMatrix_2 = sp.diags([0.5, 1.5, 2.0], 0, shape=(3, 3), format="csr")
#SparseMatrix_3=sp.diags([1.0,2.0,3.0,4.0],0,shape=(4,4),format="csr")
def test_func1():
    assert sparse_matrix.get_value(0, 1) == 1

def test_func2():
    recommendations = sparse_matrix.recommend(recommendations_vector)
    print(recommendations)
    assert recommendations[1]==12.0

def test_func3():
    #error handling for additon 
    try:
        result_2 = sparse_matrix.add_(SparseMatrix_2)
        assert result_2[2,2]==5.0
    except Exception as e:
        pytest.fail(f"Exception occurred: {e}")

def test_func4():
    x1=sparse_matrix.to_dense()
    assert x1[0][0]==0

def test_func5():
    result=sparse_matrix.non_zero_Positions()
    print(result)
    
def test_func6():
    x=sparse_matrix.List_Non_Zero()
    assert x==6.0

def test_func7():
    x=sparse_matrix.Transpose_sparse()
    assert x[1,0]==1

def test_func8():
    x=sparse_matrix.Convert_Sparse()
    assert x==2

def test_func9():
    # error handling for Multiplication of matrices 
    try:
        x=sparse_matrix.To_Multiply(SparseMatrix_2)
        assert x[2,2]==6.0
    except Exception as e:
            pytest.fail(f"Exception occurred: {e}")

