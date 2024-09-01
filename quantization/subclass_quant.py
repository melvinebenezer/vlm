import torch

class QuantizedTensor_T(torch.Tensor):

    def __new__(cls, tensor, scale, zero_point):
       
       instance = tensor.new_empty(tensor.size())
       instance.__class__ = cls
       quantized_data = torch.clamp(torch.round(tensor / scale) + zero_point, 0, 15).to(torch.uint8)

       instance = quantized_data.new_empty(quantized_data.size(), dtype=torch.uint8)
       instance.copy_(quantized_data)
       instance.scale = scale
       instance.zero_point = zero_point
       return instance

    def __init__(self, tensor, scale, zero_point):

        pass

    def dequantize(self):
        # x - zero_point * scale
        return (self.quantized_data.float() - self.zero_point) * self.scale
    
    # override the __repr__ method to show its a quantized tensor
    def __repr__(self):
        return f"QuantizedTensor_T(scale={self.scale}, zero_point={self.zero_point}, shape={tuple(self.shape)}"
    

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        args = [a.dequantize() if isinstance(a, QuantizedTensor_T) else a for a in args]

        return func(*args, **kwargs)



if __name__ == "__main__":

    # Lets see how much we save in memory

    original_tensor = torch.randn(1000, 1000) # 4MB for a 1000x1000 tensor.float32
    print(f"Memory size of original tensor: {original_tensor.element_size() * original_tensor.numel() / 1024 / 1024:.2f} MB")

    # Lets quantize the tensor
    quantized_tensor = QuantizedTensor_T(original_tensor, scale=0.1, zero_point=8)
    print(f"Memory size of quantized tensor: {quantized_tensor.element_size() * quantized_tensor.numel() / 1024 / 1024:.2f} MB")

    # qt = QuantizedTensor_T(torch.tensor([1.0, 2.0, 3.0]), scale=0.1, zero_point=0)
    # result = torch.add(qt, torch.tensor([0.5, 1.0, 1.5]))
    # print(result)
    # print(type(result))
    # print(result.__class__)