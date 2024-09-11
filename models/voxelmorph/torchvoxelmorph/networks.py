import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from ...stylegan_networks import ResBlock, StyledConv

from . import layers
from .modelio import LoadableModel, store_config_args
from torch.nn import functional as F
from einops import rearrange

def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features

class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        self.af1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std = 1e-6)
        self.do2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)
        
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = True) # Wq,Wk,Wv for each vector, thats why *3
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)
        
        self.nn1 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)        
        self.do1 = nn.Dropout(dropout)
        

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x) #gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h) # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1) #follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v) #product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)') #concat heads into one matrix, ready for next encoder block
        out =  self.nn1(out)
        out = self.do1(out)
        return out

class Transformer_orig(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attention, mlp in self.layers:
            x = attention(x, mask = mask) # go to attention
            x = mlp(x) #go to MLP_Block
        return x
        
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.feature_list = []  # Initialize an empty list to store features

        for idx in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

            # Check if the current index is a multiple of 4 (4th, 8th, ...)
            if (idx + 1) % 4 == 0:
                # Append a placeholder tensor to the feature list
                self.feature_list.append(torch.empty(0))

    def forward(self, x, mask=None):
        for idx, (attention, mlp) in enumerate(self.layers):
            x = attention(x, mask=mask)  # Go to attention
            x = mlp(x)  # Go to MLP_Block

            # Check if the current index is a multiple of 4 (4th, 8th, ...)
            if (idx + 1) % 4 == 0:
                # Append the output feature to the corresponding position in the feature list
                self.feature_list[(idx + 1) // 4 - 1] = x

        return x, self.feature_list

        
class Projector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Projector, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        print(f"In_dimension: {in_dim}")
        print(f"Out_dimension: {out_dim}")

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.fc(x)
        x = x.view(1, 64, 128, 128)  # Reshape back to 4D tensor
        return x

class VisualTransformer(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dim=128, num_tokens=8, mlp_dim=256, heads=8, depth=6,
                 emb_dropout=0.1, dropout=0.1):
        super(VisualTransformer, self).__init__()
        self.in_planes = 16
        self.L = num_tokens
        self.cT = dim
        BATCH_SIZE_TRAIN = 1

        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1, bias=False)  # Updated input channels to 6
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.dlayer1 = self._decoder_layer(64, 32, 512, 512)
        self.dlayer2 = self._decoder_layer(32, 32, 512, 512)
        self.dlayer3 = self._decoder_layer(32, 16, 512, 512)

        # Tokenization parameters
        self.token_wA = nn.Parameter(torch.empty(BATCH_SIZE_TRAIN, self.L, 64), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(BATCH_SIZE_TRAIN, 64, self.cT), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.projector = Projector(9 * 16, 64 * 128 * 128)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.segmentation_conv = nn.Conv2d(
            in_channels=16,  # dim is the hidden size from your transformer
            out_channels=16,   # assuming you want 16 segmentation channels
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Define upsample layer
        self.upsample = nn.Upsample(
            scale_factor=2,  # adjust scale_factor based on your needs
            mode='nearest'  # or 'bilinear', depending on your preference
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
        
    def _decoder_layer(self, in_channels, out_channels, height, width):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Upsample(size=(height, width), mode='bilinear', align_corners=False)
        )

    def forward(self, img, mask=None):
        x = F.relu(self.bn1(self.conv1(img)))
        #print(f'Shape of x before layer1: {x.shape}')
        x = self.layer1(x)
        #print(f'Shape of x after layer1: {x.shape}')
        x = self.layer2(x)
        #print(f'Shape of x after layer2: {x.shape}')
        x = self.layer3(x)
        #print(f'Shape of x after layer3: {x.shape}')
        skip=x

        x = rearrange(x, 'b c h w -> b (h w) c')
        #print(f'Shape of x after rearrangement: {x.shape}')

        # Tokenization
        wa = rearrange(self.token_wA, 'b h w -> b w h')
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        #print(f'Shape of x after tokenization: {x.shape}')
        x += self.pos_embedding
        #print(f'Shape of x after positional embedding: {x.shape}')
        x = self.dropout(x)
        #print(f'Shape of x after dropout: {x.shape}')
        x = self.transformer(x, mask)
        #print(f'Shape of x after transformer: {x.shape}')
        x =x.view(1,16,512,512)
        x = self.segmentation_conv(x)
        print(f'Shape of x after segmentation: {x.shape}')
        x = self.upsample(x)
        print(f'Shape of x after upsample: {x.shape}')
        #x=self.projector(x)
        #print(f'Shape of x after projector: {x.shape}')
        #x = skip+x
        #print(f'Shape of x after concatenation: {x.shape}')
        #x = self.dlayer1(x)
        #print(f'Shape of x after layer1: {x.shape}')
        #x = self.dlayer2(x)
        #print(f'Shape of x after layer2: {x.shape}')
        #x = self.dlayer3(x)
        #print(f'Shape of x after layer3: {x.shape}')
        #x = self.to_cls_token(x[:, 0])
        #print(f'Shape of x after class tokens: {x.shape}')
        #x = self.nn1(x)
        #print(f'Shape of x after nn1: {x.shape}')

        return x
        
        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        #print(out.size())
        return out

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
        
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)        
        

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class VxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        
        
        
        self.visualtransformermodel = VisualTransformer(BasicBlock, num_blocks=[3, 3, 3], num_classes=16, dim=16, num_tokens=8, mlp_dim=32, heads=8, depth=6,
                  emb_dropout=0.1, dropout=0.1)

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )
        
        

        # configure unet to flow field layer
        # 最终都会变成2维度的flow图，也就是形变场
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)
        #print(f"Size of x before flow layer: {x.shape}")

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)
        #print(f"Size of x before preint flow: {pos_flow.shape}")

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None
        #print(f"Size of x before integrate layer: {neg_flow.shape}")

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            #print(f"Size of x after pos_flow: {pos_flow.shape}")
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, pos_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow

    def predict(self, image, flow, svf=True, **kwargs):

        if svf:
            flow = self.integrate(flow)

            if self.fullsize:
                flow = self.fullsize(flow)

        return self.transformer(image, flow, **kwargs)

    def get_flow_field(self, flow_field):
        if self.integrate:
            flow_field = self.integrate(flow_field)

            # resize to final resolution
            if self.fullsize:
                flow_field = self.fullsize(flow_field)

        return flow_field
        


class BiGRU_VERT_HORIZ(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, batch_first=True, dropout=0.0):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=batch_first, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional
        self.squash = nn.Softmax(dim=-1)

    def forward(self, x, h_prev=None):
        # Initialize an empty tensor to store the results
        output = torch.empty((x.size(0), x.size(2), 64))

        # Apply BiGRU to each feature separately
        for i in range(x.size(2)):
            feature = x[:, :, i].unsqueeze(-1)
            out, h_prev = self.gru(feature, h_prev if h_prev is not None else None)
            out = torch.cat((out[:, -1, :out.size(2)//2], out[:, 0, out.size(2)//2:]), dim=1)
            out = nn.ReLU()(self.fc(out))
            out = self.squash(out)  # Apply squash layer
            output[:, i, :] = out

        return output

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, batch_first=True, dropout=0.0):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=batch_first, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional
        self.squash = nn.Softmax(dim=-1)

    def forward(self, x):
        # Initialize an empty tensor to store the results
        output = torch.empty((x.size(0), x.size(2), 64))

        # Apply BiGRU to each feature separately
        for i in range(x.size(2)):
            feature = x[:, :, i].unsqueeze(-1)
            out, _ = self.gru(feature)
            out = torch.cat((out[:, -1, :out.size(2)//2], out[:, 0, out.size(2)//2:]), dim=1)
            out = nn.ReLU()(self.fc(out))
            out=self.squash(out)
            output[:, i, :] = out

        return output
        
class UnetDFMIR(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
        '''
        nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
        '''

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 6
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 6
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf
 
    def forward(self, x):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x
                
class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
        '''
        nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
        '''
        BATCH_SIZE_TRAIN = 1
        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features
            
        self.L = 8
        self.cT = 128

        self.upsamplen = nn.Upsample(scale_factor=2, mode='nearest')
        # Tokenization parameters
        self.token_wA = nn.Parameter(torch.empty(BATCH_SIZE_TRAIN, self.L, 64), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(BATCH_SIZE_TRAIN, 64, self.cT), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (8 + 1), 128))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 128))
        self.dropout = nn.Dropout(0.1)

        self.transformer = Transformer_orig(128, 6, 8, 256, 0.1)


        # configure encoder (down-sampling path)
        prev_nf = 6
        self.downarm = nn.ModuleList()
        #print("Number of dimensions: ",ndims)
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 6
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf
        self.bi_gru = BiGRU(input_size=64, hidden_size=128, output_size=64)
            
    # Upsampling block
        model = [nn.ConvTranspose2d(128, 64,
                                    kernel_size=6, stride=2,
                                    padding=1, output_padding=0,
                                    bias=False),
                     nn.BatchNorm2d(64),
                     nn.ReLU(True)]
        
        setattr(self, 'upsample', nn.Sequential(*model))
 
    def forward(self, img, mask=None):
        #print(f'Shape of x before unet: {x.shape}')
        # get encoder activations
        x_enc = [img]
        #print(f'Shape of x input: {x_enc[-1].shape}')
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))
            #print(f'Shape of x after layer: {x_enc[-1].shape}')
            
        #print(f"Size of x after downarm: {x_enc[-1].shape}")
        x = x_enc.pop()
        #print(f'Shape of x after popping: {x.shape}')
        x = rearrange(x, 'b c h w -> b (h w) c')   #(batch_size, sequence_length, input_size)
        #print(f'Shape of x after rearrangement: {x.shape}')
        x=self.bi_gru(x)
        #print(f'Shape of x after BGRU layer : {x.shape}')
        x = x.to('cuda')
    
        # Tokenization
        wa = rearrange(self.token_wA, 'b h w -> b w h')
        wa = wa.to('cuda')
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)

        cls_tokens = self.cls_token.expand(x_enc[-1].shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        #print(f'Shape of x after tokenization: {x.shape}')
        x += self.pos_embedding
        #print(f'Shape of x after positional embedding: {x.shape}')
        x = self.dropout(x)
        #print(f'Shape of x after dropout: {x.shape}')
        x = self.transformer(x, mask)
        #print(f'Shape of x after transformer: {x.shape}')
        #x=self.dropout(x)
        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        #print(f'Shape of x after transformer output reshaping: {x.shape}')
            # upsample transformer output
        x = self.upsample(x)
        #print(f'Shape of x after upsampling: {x.shape}')
            # concat transformer output and resnet output
        #x = torch.cat([transformer_out, x], dim=1)
            # channel compression
            #x = self.cc(x)
        # conv, upsample, concatenate series
        
        
        for layer in self.uparm:
            x = layer(x)
            #print(f'Shape of x after convolution in uparm: {x.shape}')
            x = self.upsamplen(x)
            #print(f'Shape of x after nn.upsampling in uparm: {x.shape}')
            temp=x_enc.pop()
            #print(f'Shape of x_encoder from last layer: {temp.shape}')
            x = torch.cat([x, temp], dim=1)
            #print(f'Shape of x after concatenation: {x.shape}')
            
        

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)
            #print(f'Shape of x after extra layer (conv): {x.shape}')
        return x

class ConvBlock2(nn.Module):
    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

from util.trans_model import *
class Unet_Transformer(nn.Module):
    def __init__(self, inshape, config, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
        '''
        nb_features = [
            [16, 32, 32, 64, 64, 64],  # encoder
            [64, 64, 64, 32, 32, 32, 16]  # decoder
        ]
        '''

        # build feature list automatically
        self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 1
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock2(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        prev_nf2 = 1
        self.downarm2 = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm2.append(ConvBlock2(ndims, prev_nf2, nf, stride=2))
            prev_nf2 = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i]*2 if i > 0 else prev_nf*2
            self.uparm.append(ConvBlock2(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock2(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

        self.config = config

        self.avgpool = nn.AdaptiveAvgPool2d((self.config.vert_anchors, self.config.horz_anchors))

        # self.image_encoder = LidarEncoder(512, in_channels=1)
        # self.lidar_encoder = LidarEncoder(num_classes=512, in_channels=1)

        self.transformer1 = GPT(n_embd=16,
                                n_head=config.n_head,#4
                                block_exp=config.block_exp,#4
                                n_layer=config.n_layer,#8
                                vert_anchors=config.vert_anchors,#8
                                horz_anchors=config.horz_anchors,#8
                                seq_len=config.seq_len,#1
                                embd_pdrop=config.embd_pdrop,#0.1
                                attn_pdrop=config.attn_pdrop,#0.1
                                resid_pdrop=config.resid_pdrop,#0.1
                                config=config)
        self.transformer2 = GPT(n_embd=32,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer3 = GPT(n_embd=32,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer4 = GPT(n_embd=64,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer5 = GPT(n_embd=64,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)

        self.trans_list = \
            [self.transformer1, self.transformer2, self.transformer3, self.transformer4, self.transformer5]
        #self.fuse_list = nn.ModuleList()
        #self.fuse_list.append(nn.Conv2d(16*2,16,1,1))
        #self.fuse_list.append(nn.Conv2d(32 * 2, 32, 1, 1))
        #self.fuse_list.append(nn.Conv2d(32 * 2, 32, 1, 1))
        #self.fuse_list.append(nn.Conv2d(64 * 2, 64, 1, 1))
        #self.fuse_list.append(nn.Conv2d(64 * 2, 64, 1, 1))



    def forward(self, x, y):
        import math
        # get encoder activations
        x_enc = [x]
        y_enc = [y]
        xy_fuse = [torch.cat([x,y],dim=1)]
        for i,layer in enumerate(self.downarm):
            tmp = layer(x_enc[-1])
            tmp2 = self.downarm2[i](y_enc[-1])
            #print(tmp.size())
            #print(tmp2.size())

            image_embd_layer1 = self.avgpool(tmp)
            lidar_embd_layer1 = self.avgpool(tmp2)
            image_features_layer1, lidar_features_layer1 = self.trans_list[i](image_embd_layer1, lidar_embd_layer1, None)
            image_features_layer1 = F.interpolate(image_features_layer1, scale_factor=16/int(math.pow(2,i)), mode='bilinear')
            lidar_features_layer1 = F.interpolate(lidar_features_layer1, scale_factor=16/int(math.pow(2,i)), mode='bilinear')
            tmp = tmp + image_features_layer1
            tmp2 = tmp2 + lidar_features_layer1

            #torch.cat([tmp,tmp2],dim=1)
            x_enc.append(tmp)
            y_enc.append(tmp2)
            #xy_fuse.append(self.fuse_list[i](torch.cat([tmp,tmp2],dim=1)))
            xy_fuse.append(torch.cat([tmp,tmp2],dim=1))


        # conv, upsample, concatenate series
        x = xy_fuse.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, xy_fuse.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x

class Whole_Transformer(nn.Module):
    def __init__(self, inshape, config, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
        '''
        nb_features = [
            [16, 32, 32, 64, 64, 64],  # encoder
            [64, 64, 64, 32, 32, 32, 16]  # decoder
        ]
        '''

        # build feature list automatically
        self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 1
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock2(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        prev_nf2 = 1
        self.downarm2 = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm2.append(ConvBlock2(ndims, prev_nf2, nf, stride=2))
            prev_nf2 = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock2(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock2(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

        self.config = config

        self.avgpool = nn.AdaptiveAvgPool2d((self.config.vert_anchors, self.config.horz_anchors))

        self.transformer1 = GPT(n_embd=16,
                                n_head=config.n_head,#4
                                block_exp=config.block_exp,#4
                                n_layer=config.n_layer,#8
                                vert_anchors=config.vert_anchors,#8
                                horz_anchors=config.horz_anchors,#8
                                seq_len=config.seq_len,#1
                                embd_pdrop=config.embd_pdrop,#0.1
                                attn_pdrop=config.attn_pdrop,#0.1
                                resid_pdrop=config.resid_pdrop,#0.1
                                config=config)
        self.transformer2 = GPT(n_embd=32,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer3 = GPT(n_embd=64,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer4 = GPT(n_embd=64,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer5 = GPT(n_embd=64,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)

        self.transformer6 = GPT(n_embd=64,
                                n_head=config.n_head,#4
                                block_exp=config.block_exp,#4
                                n_layer=config.n_layer,#8
                                vert_anchors=config.vert_anchors,#8
                                horz_anchors=config.horz_anchors,#8
                                seq_len=config.seq_len,#1
                                embd_pdrop=config.embd_pdrop,#0.1
                                attn_pdrop=config.attn_pdrop,#0.1
                                resid_pdrop=config.resid_pdrop,#0.1
                                config=config)
        self.transformer7 = GPT(n_embd=64,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer8 = GPT(n_embd=32,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer9 = GPT(n_embd=16,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        # self.transformer10 = GPT(n_embd=16,
        #                         n_head=config.n_head,
        #                         block_exp=config.block_exp,
        #                         n_layer=config.n_layer,
        #                         vert_anchors=config.vert_anchors,
        #                         horz_anchors=config.horz_anchors,
        #                         seq_len=config.seq_len,
        #                         embd_pdrop=config.embd_pdrop,
        #                         attn_pdrop=config.attn_pdrop,
        #                         resid_pdrop=config.resid_pdrop,
        #                         config=config)

        self.trans_list = \
            [self.transformer1, self.transformer2, self.transformer3, self.transformer4, self.transformer5]
        self.trans_list_skip = \
            [self.transformer6, self.transformer7, self.transformer8, self.transformer9]

        self.fuse_list = nn.ModuleList()
        self.fuse_list.append(nn.Conv2d(16*2,16,1,1))
        self.fuse_list.append(nn.Conv2d(32 * 2, 32, 1, 1))
        self.fuse_list.append(nn.Conv2d(64 * 2, 64, 1, 1))
        self.fuse_list.append(nn.Conv2d(64 * 2, 64, 1, 1))
        self.fuse_list.append(nn.Conv2d(64 * 2, 64, 1, 1))


    def forward(self, x, y):
        import math
        # get encoder activations
        x_enc = [x]
        y_enc = [y]
        xy_fuse = [torch.cat([x,y],dim=1)]
        for i,layer in enumerate(self.downarm):
            tmp = layer(x_enc[-1])
            tmp2 = self.downarm2[i](y_enc[-1])
            #print(tmp.size())
            #print(tmp2.size())

            image_embd_layer1 = self.avgpool(tmp)
            lidar_embd_layer1 = self.avgpool(tmp2)
            image_features_layer1, lidar_features_layer1 = self.trans_list[i](image_embd_layer1, lidar_embd_layer1, None)
            image_features_layer1 = F.interpolate(image_features_layer1, scale_factor=16/int(math.pow(2,i)), mode='bilinear')
            lidar_features_layer1 = F.interpolate(lidar_features_layer1, scale_factor=16/int(math.pow(2,i)), mode='bilinear')
            tmp = tmp + image_features_layer1
            tmp2 = tmp2 + lidar_features_layer1

            #torch.cat([tmp,tmp2],dim=1)
            x_enc.append(tmp)
            y_enc.append(tmp2)
            xy_fuse.append(self.fuse_list[i](torch.cat([tmp,tmp2],dim=1)))


        # conv, upsample, concatenate series
        x = xy_fuse.pop()
        for i,layer in enumerate(self.uparm):
            x = layer(x)
            x = self.upsample(x)
            skip_feat = xy_fuse.pop()
            # print(x.size())
            # print(skip_feat.size())
            if i<len(self.trans_list_skip):
                image_embd_layer1 = self.avgpool(x)
                lidar_embd_layer1 = self.avgpool(skip_feat)
                image_features_layer1, lidar_features_layer1 = self.trans_list_skip[i](image_embd_layer1, lidar_embd_layer1, None)
                image_features_layer1 = F.interpolate(image_features_layer1, scale_factor=2* int(math.pow(2, i)),
                                                      mode='bilinear')
                lidar_features_layer1 = F.interpolate(lidar_features_layer1, scale_factor=2* int(math.pow(2, i)),
                                                      mode='bilinear')
                x = x + image_features_layer1
                skip_feat = skip_feat + lidar_features_layer1
            x = torch.cat([x, skip_feat], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x

class DualUnet(nn.Module):
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
        '''
        nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
        '''

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            # 通常情况下，enc_nf取第一行，dec_nf取第二行
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        # 输入的xy图像叠加到一起，因此是两个channel的
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
        # 最后，prev_nf就变成了最终的encoder输出的channel

        # self.conv_combine = ConvBlock(ndims, prev_nf*2,prev_nf, stride=1)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            # 解码器和编码器保持同样的深度
            channels = prev_nf + enc_history[i]*2 if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf = prev_nf*2 + 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

        self.fusion = nn.ModuleList()
        self.fusion.append(nn.Conv2d(256*2,256,1,1))
        self.fusion.append(nn.Conv2d(128*2,128,1,1))
        self.fusion.append(nn.Conv2d(64*2,64,1,1))

    def forward_(self, x, x_enc2):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))
        # 输出结果的排列顺序：以 [16, 32, 64, 128, 256],  # encoder为例
        # 2 -> 16 -> 32 -> 64 ->128 -> 256 共5次卷积
        # 224*224*2 -> 112*112*16 -> 56*56*32 -> 28*28*64 -> 14*14*128 -> 7*7*256
        # 能给到的feature map为 224*224*2, 224*224*64, 112*112*128, 56*56*256,

        # [128, 256, 256]  # encoder
        # [256, 128, 64, 16, 8]  # decoder
        # conv, upsample, concatenate series
        x = x_enc.pop()
        # x2 = x_enc2.pop()
        # x = self.conv_combine(torch.cat([x,x2],dim=1))
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop(), x_enc2.pop()], dim=1)

        # decoder   [128, 64, 32, 16, 16, 8]  # decoder
        # 这里的变化为：从7*7*256变为14*14* 128（叠加第1次，）->64(叠加1次) -> 32(叠加一次) 16 (叠加一次)， 最后一次叠加是224*224*2和224*224*16的叠加，没有卷积了
        # 1. 7*7*256 -> 14*14*128, 而且 把concat操作完成了 14*14*256
        # 2. 14*14*64 -> 28*28*64
        # 224*224*16
        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)
        # 最后多一个8，就是224*224*16+2进去，输出224*224*8

        return x

    def forward(self, x, x_enc2, x_enc3):
        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))
        # 出来的是 224*224*2, 112*112*128, 56*56*256, 28*28*256
        # 能给到的feature map为 224*224*2, 224*224*64, 112*112*128, 56*56*256

        # [128, 256, 256]  # encoder
        # [256, 128, 64, 16, 8]  # decoder
        # conv, upsample, concatenate series
        x = x_enc.pop()
        # x2 = x_enc2.pop()
        # x = self.conv_combine(torch.cat([x,x2],dim=1))
        for i, layer in enumerate(self.uparm):
            x = layer(x)
            x = self.upsample(x)
            x_enc_fused = self.fusion[i](torch.cat([x_enc2.pop(),x_enc3.pop()],dim=1))
            x = torch.cat([x, x_enc.pop(), x_enc_fused], dim=1)

        # decoder   [128, 64, 32, 16, 16, 8]  # decoder
        # 这里的变化为：从7*7*256变为14*14* 128（叠加第1次，）->64(叠加1次) -> 32(叠加一次) 16 (叠加一次)， 最后一次叠加是224*224*2和224*224*16的叠加，没有卷积了
        # 1. 7*7*256 -> 14*14*128, 而且 把concat操作完成了 14*14*256
        # 2. 14*14*64 -> 28*28*64
        # 224*224*16
        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x
        # 最后多一个8，就是224*224*16+2进去，输出224*224*8


class AttentionNet(nn.Module):
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            pass
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            # 通常情况下，enc_nf取第一行，dec_nf取第二行
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        # 输入的xy图像叠加到一起，因此是两个channel的
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
        # 最后，prev_nf就变成了最终的encoder输出的channel

        # self.conv_combine = ConvBlock(ndims, prev_nf*2,prev_nf, stride=1)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            # 解码器和编码器保持同样的深度
            channels = prev_nf + enc_history[i]*2 if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf = prev_nf*2 + 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

        self.fusion = nn.ModuleList()
        # self.fusion.append(nn.Conv2d(256*2,256,1,1))
        # self.fusion.append(nn.Conv2d(128*2,128,1,1))
        # self.fusion.append(nn.Conv2d(64*2,64,1,1))
        self.fusion.append(NLBlockND_cross(256, dimension=2))
        #self.fusion.append(nn.Conv2d(128 * 2, 128, 1, 1))
        self.fusion.append(NLBlockND_cross(128, dimension=2))
        self.fusion.append(nn.Conv2d(64*2,64,1,1))
        #self.fusion.append(NLBlockND_cross(128, dimension=2))
        #self.fusion.append(NLBlockND_cross(64, dimension=2))
        self.activation_atten = nn.LeakyReLU(0.2)

    def forward(self, x, x_enc2, x_enc3):
        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))
        # 出来的是 224*224*2, 112*112*128, 56*56*256, 28*28*256
        # 能给到的feature map为 224*224*2, 224*224*64, 112*112*128, 56*56*256


        x = x_enc.pop()
        # x2 = x_enc2.pop()
        # x = self.conv_combine(torch.cat([x,x2],dim=1))
        for i, layer in enumerate(self.uparm):
            x = layer(x)
            x = self.upsample(x)
            tmp1 = x_enc2.pop()
            tmp2 = x_enc3.pop()
            if i ==0 or i == 1:
                x_enc_atten1 = self.activation_atten(self.fusion[i](tmp1,tmp2))
                x_enc_atten2 = self.activation_atten(self.fusion[i](tmp2,tmp1))
                x_enc_fused = torch.cat([x_enc_atten1, x_enc_atten2],dim=1)
            else:
                x_enc_fused = self.fusion[i](torch.cat([tmp1, tmp2], dim=1))
            x = torch.cat([x, x_enc.pop(), x_enc_fused], dim=1)
        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x
        # 最后多一个8，就是224*224*16+2进去，输出224*224*8


class DecoderNet(nn.Module):
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()


        # build feature list automatically
        if isinstance(nb_features, int):
            pass
            # if nb_levels is None:
            #     raise ValueError('must provide unet nb_levels if nb_features is an integer')
            # feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            # self.enc_nf = feats[:-1]
            # self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            # 通常情况下，enc_nf取第一行，dec_nf取第二行
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        # 输入的xy图像叠加到一起，因此是两个channel的
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
        # 最后，prev_nf就变成了最终的encoder输出的channel

        # self.conv_combine = ConvBlock(ndims, prev_nf*2,prev_nf, stride=1)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            # 解码器和编码器保持同样的深度
            channels = prev_nf + enc_history[i]*2 if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf = prev_nf*2 + 64
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

        # self.fusion = nn.ModuleList()
        # self.fusion.append(nn.Conv2d(256*2,256,1,1))
        # self.fusion.append(nn.Conv2d(128*2,128,1,1))
        # self.fusion.append(nn.Conv2d(64*2,64,1,1))
        self.conv1 = ConvBlock(ndims, 256*2,256,stride=1)
        self.conv2 = ConvBlock(ndims, 256,256,stride=2)

    def forward(self, x_enc1, x_enc2):

        out1 = self.conv1(torch.cat([x_enc1[-1],x_enc2[-1]], dim=1))
        x = self.conv2(out1)

        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc1.pop(), x_enc2.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)
        # 最后多一个8，就是224*224*16+2进去，输出224*224*8

        return x



class VxmAttentionNet(LoadableModel):

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):

        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = AttentionNet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        # 最终都会变成2维度的flow图，也就是形变场
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, enc, enc3, registration=False):

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        # x = self.unet_model(x, enc)
        x = self.unet_model(x, enc, enc3)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow


class VxmDecoderDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = DecoderNet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        # 最终都会变成2维度的flow图，也就是形变场
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, enc1, enc2, registration=False):

        # concatenate inputs and propagate unet
        #x = torch.cat([source, target], dim=1)
        x = self.unet_model(enc1,enc2)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow




class VxmDenseTransformer(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        config = GlobalConfig()

        # configure core unet model
        self.unet_model = Unet_Transformer(
            inshape,
            config,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        ).to('cuda')

        # configure unet to flow field layer
        # 最终都会变成2维度的flow图，也就是形变场
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        #x = torch.cat([source, target], dim=1)
        x = self.unet_model(source,target)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, pos_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow





class VxmDenseTransformerWhole(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        config = GlobalConfig()

        # configure core unet model
        self.unet_model = Whole_Transformer(
            inshape,
            config,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        ).to('cuda')

        # configure unet to flow field layer
        # 最终都会变成2维度的flow图，也就是形变场
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        #x = torch.cat([source, target], dim=1)
        x = self.unet_model(source,target)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, pos_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow

class VxmDenseDual(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = DualUnet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        # 最终都会变成2维度的flow图，也就是形变场
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, enc, enc3, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        # x = self.unet_model(x, enc)
        x = self.unet_model(x, enc, enc3)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out
