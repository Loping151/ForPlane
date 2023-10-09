import tinycudann as tcnn
import torch 

encoder = tcnn.Encoding(
                n_input_dims=4,
                encoding_config={
                    "otype": "OneBlob",
                    "nbins": 2,
                },
            )

x = torch.rand(100,4).cuda()

y = encoder(x)