import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z=np.array(z)
        z=z-(np.max(z))
        exp_z=np.exp(z)
        softmax=exp_z/np.sum(exp_z)


        return np.round(softmax,4)
        pass


#         import numpy as np

# class Solution:
#     def softmax(self, z):
#         # Step 1: convert to numpy array (safety)
#         z = np.array(z)
        
#         # Step 2: numerical stability
#         z = z - np.max(z)
        
#         # Step 3: exponentiation
#         exp_z = np.exp(z)
        
#         # Step 4: normalize
#         softmax = exp_z / np.sum(exp_z)
        
#         # Step 5: round to 4 decimals
#         return np.round(softmax, 4)
