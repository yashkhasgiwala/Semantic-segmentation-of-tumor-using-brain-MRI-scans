from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def Unet(dims):
    inp = Input(dims)

    conv_1 = Conv2D(filters = 64, kernel_size = 3, padding= 'same', activation= 'relu')(inp)
    conv_1 = Conv2D(filters = 64, kernel_size = 3, padding= 'same', activation= 'relu')(conv_1)
    max_pool1 = MaxPool2D(pool_size=(2,2))(conv_1)

    conv_2 = Conv2D(filters = 128, kernel_size = 3, padding= 'same', activation= 'relu')(max_pool1)
    conv_2 = Conv2D(filters = 128, kernel_size = 3, padding= 'same', activation= 'relu')(conv_2)
    bn_2 = BatchNormalization(axis=3)(conv_2)
    max_pool2 = MaxPool2D(pool_size=(2,2))(bn_2)

    conv_3 = Conv2D(filters = 256, kernel_size = 3, padding= 'same', activation= 'relu')(max_pool2)
    conv_3 = Conv2D(filters = 256, kernel_size = 3, padding= 'same', activation= 'relu')(conv_3)
    max_pool3 = MaxPool2D(pool_size=(2,2))(conv_3)
    
    conv_4 = Conv2D(filters = 512, kernel_size = 3, padding= 'same', activation= 'relu')(max_pool3)
    conv_4 = Conv2D(filters = 512, kernel_size = 3, padding= 'same', activation= 'relu')(conv_4)
    bn_4 = BatchNormalization(axis=3)(conv_4)
    max_pool4 = MaxPool2D(pool_size=(2,2))(bn_4)

    conv_5 = Conv2D(filters = 1024, kernel_size = 3, padding= 'same', activation= 'relu')(max_pool4)
    conv_5 = Conv2D(filters = 1024, kernel_size = 3, padding= 'same', activation= 'relu')(conv_5)
    drop_5 = Dropout(0.3)(conv_5)

    up_6 = Conv2DTranspose(filters = 512,kernel_size= 3, strides=(2, 2), padding="same")(drop_5)
    concat_6 = concatenate([up_6,bn_4], axis=3)
    conv_6 = Conv2D(filters = 512, kernel_size = 3, padding= 'same', activation= 'relu')(concat_6)
    conv_6 = Conv2D(filters = 512, kernel_size = 3, padding= 'same', activation= 'relu')(conv_6)
    bn_6 = BatchNormalization(axis=3)(conv_6)

    up_7 = Conv2DTranspose(filters = 256,kernel_size= 3, strides=(2, 2), padding="same")(bn_6)
    concat_7 = concatenate([up_7,conv_3], axis=3)
    conv_7 = Conv2D(filters = 256, kernel_size = 3, padding= 'same', activation= 'relu')(concat_7)
    conv_7 = Conv2D(filters = 256, kernel_size = 3, padding= 'same', activation= 'relu')(conv_7)

    up_8 = Conv2DTranspose(filters = 128,kernel_size= 3, strides=(2, 2), padding="same")(conv_7)
    concat_8 = concatenate([up_8,bn_2], axis=3)
    conv_8 = Conv2D(filters = 128, kernel_size = 3, padding= 'same', activation= 'relu')(concat_8)
    conv_8 = Conv2D(filters = 128, kernel_size = 3, padding= 'same', activation= 'relu')(conv_8)
    bn_8 = BatchNormalization(axis=3)(conv_8)

    up_9 = Conv2DTranspose(filters = 64,kernel_size= 3, strides=(2, 2), padding="same")(bn_8)
    concat_9 = concatenate([up_9,conv_1], axis=3)
    conv_9 = Conv2D(filters = 64, kernel_size = 3, padding= 'same', activation= 'relu')(concat_9)
    conv_9 = Conv2D(filters = 64, kernel_size = 3, padding= 'same', activation= 'relu')(conv_9)

    conv_10 = Conv2D(filters = 1, kernel_size = 1, activation= 'sigmoid')(conv_9)

    return Model(inputs = inp, outputs = conv_10)

#Source - https://github.com/nikhilroxtomar/Deep-Residual-Unet 

def bn_act(x, act=True):
    x = BatchNormalization()(x)
    if act == True:
        x = Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = UpSampling2D((2, 2))(x)
    c = Concatenate()([u, xskip])
    return c

def ResUNet():
    f = [16, 32, 64, 128, 256]
    inputs = Input((256, 256, 3))
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    
    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])
    
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])
    
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])
    
    outputs = Conv2D(filters = 1, kernel_size = 1, activation="sigmoid")(d4)
    model = Model(inputs, outputs)
    return model



    

    