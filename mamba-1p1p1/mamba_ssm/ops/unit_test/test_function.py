import unittest
import torch
import torch.nn.functional as F

def compute_middle_attn_vector(delta, delta_bias, A, B, C, L, x_shape, cls_first="middle", full_vector=False):
    dt = F.softplus(delta + delta_bias.unsqueeze(0).unsqueeze(-1))
    dA = torch.exp(torch.einsum("bdl,dn->bldn", dt, A)) ### or dt@A!!
    dB = torch.einsum("bdl,bnl->bldn", dt, B.squeeze(1))
    AttnVecorOverCLS = torch.zeros(x_shape).to(dA.device) #BHL: L vectors per batch and channel
    cls_pos = 98
    for t in range(cls_pos):
        curr_C = C[:,:,:,cls_pos]
        currA = torch.ones(dA.shape[0],dA.shape[2],dA.shape[3]).to(dA.device)
        if t < (cls_pos-1):
            for i in range(cls_pos-1-t):
                currA = currA*dA[:,cls_pos-1-i,:,:]
        currB = dB[:,t,:,:]
        AttnVecorOverCLS[:,:,t] = torch.sum(curr_C*currA*currB, axis=-1) 
    return AttnVecorOverCLS



# This function is the same as compute_middle_attn_vector, but Curr_C has -1 to extract the last element.
def compute_middle_attn_vector_2(delta, delta_bias, A, B, C, L, x_shape, cls_first="middle", full_vector=False):
    dt = F.softplus(delta + delta_bias.unsqueeze(0).unsqueeze(-1))
    dA = torch.exp(torch.einsum("bdl,dn->bldn", dt, A)) ### or dt@A!!
    dB = torch.einsum("bdl,bnl->bldn", dt, B.squeeze(1))
    AttnVecorOverCLS = torch.zeros(x_shape).to(dA.device) #BHL: L vectors per batch and channel
    cls_pos = L
    for t in range(cls_pos):
        curr_C = C[:,:,:,cls_pos-1] #  NOTE: -1 to extract the last element
        currA = torch.ones(dA.shape[0],dA.shape[2],dA.shape[3]).to(dA.device)
        if t < (cls_pos-1):
            for i in range(cls_pos-1-t):
                currA = currA*dA[:,cls_pos-1-i,:,:]
        currB = dB[:,t,:,:]
        AttnVecorOverCLS[:,:,t] = torch.sum(curr_C*currA*currB, axis=-1) 
    return AttnVecorOverCLS


def compute_middle_attn_vector_no_loops(delta, delta_bias, A, B, C, L, x_shape, cls_position=None, full_vector=False):

    dt = F.softplus(delta + delta_bias.unsqueeze(0).unsqueeze(-1))
    dA = torch.exp(torch.einsum("bdl,dn->bldn", dt, A))
    dB = torch.einsum("bdl,bnl->bldn", dt, B.squeeze(1))
    
    AttnVecorOverCLS = torch.zeros(x_shape).to(dA.device) #BHL: L vectors per batch and channel
    
    if full_vector:
        
        # -1 to extract the last element
        cls_pos = L - 1
        curr_C = C[:,:,:,cls_pos]

        # Compute the product of all dA except the last element.
        curr_A = torch.flip(dA, [1])
        curr_A = torch.cumprod(curr_A, dim=1)
        curr_A = torch.flip(curr_A, [1])
        curr_A = curr_A[:,1:cls_pos+1,:,:]
        
        # Compute the product of all dB except the last element.
        curr_B = dB[:,:cls_pos,:,:]

        AttnVecorOverCLS[:,:,:cls_pos] = torch.sum(curr_C*curr_A*curr_B, axis=-1).transpose(1,2)

        # Compute the last element
        curr_B = dB[:,cls_pos,:,:]
        AttnVecorOverCLS[:,:,cls_pos] = torch.sum(curr_C*curr_B, axis=-1)

    else:
        
        assert cls_position is not None, "cls_position must be provided when full_vector is False"

        cls_pos = cls_position

        # Extact C at cls position.
        curr_C = C[:,:,:,cls_pos]

        dA = dA[:,1:cls_pos,:,:]
        curr_A = torch.flip(dA, [1])
        curr_A = torch.cumprod(curr_A, dim=1)
        curr_A = torch.flip(curr_A, [1])
        
        # Extract dB at up to cls_pos-1
        curr_B = dB[:,:cls_pos-1,:,:]

        # Compute the product of all dA[:,cls_pos-1-i,:,:] for i in range(cls_pos-1)
        AttnVecorOverCLS[:,:,:cls_pos-1] = torch.sum(curr_C*curr_A*curr_B, axis=-1).transpose(1,2)

        # Compute the cls_pos element
        curr_B = dB[:,cls_pos-1,:,:]
        AttnVecorOverCLS[:,:,cls_pos-1] = torch.sum(curr_C*curr_B, axis=-1)

    return AttnVecorOverCLS


class TestFun(unittest.TestCase):

    def test_vector_upto_cls(self):
        
        #B = 1, n = 16, d = 4, L = 100

        delta = torch.randn(1, 4, 100).abs()
        delta_bias = torch.randn(4).abs()
        A = torch.randn(4, 16).abs()
        B = torch.randn(1, 16, 100).abs()
        C = torch.randn(1, 1, 16, 100).abs()
        L = 100
        x_shape = (1, 4, 100)
        cls_position = 98
        full_vector = False
        expected = compute_middle_attn_vector(delta, delta_bias, A, B, C, L, x_shape)
        result = compute_middle_attn_vector_no_loops(delta, delta_bias, A, B, C, L, x_shape, cls_position=cls_position, full_vector=full_vector)

        self.assertTrue(torch.allclose(expected, result, atol=1e-6))
    
    def test_full_vector(self):
            
            # B = 1, n = 16, d = 4, L = 100
    
            delta = torch.randn(1, 4, 100).abs()
            delta_bias = torch.randn(4).abs()
            A = torch.randn(4, 16).abs()
            B = torch.randn(1, 16, 100).abs()
            C = torch.randn(1, 1, 16, 100).abs()
            L = 100
            x_shape = (1, 4, 100) 
            full_vector = True

            expected = compute_middle_attn_vector_2(delta, delta_bias, A, B, C, L, x_shape)
            result = compute_middle_attn_vector_no_loops(delta, delta_bias, A, B, C, L, x_shape, cls_position=None, full_vector=full_vector)
    
            self.assertTrue(torch.allclose(expected, result, atol=1e-6))

if __name__ == "__main__":
    unittest.main()
