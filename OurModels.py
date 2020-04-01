def LinearModel(nCategories, samplingrate=16000,
                      inputLength=16000):
    # simple LSTM
    sr = samplingrate
    iLen = inputLength
    inputs = L.Input((inputLength,), name='input')

    x = L.Reshape((1, -1))(inputs)

    m = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, iLen),
                       padding='same', sr=sr, n_mels=80,
                       fmin=40.0, fmax=sr / 2, power_melgram=1.0,
                       return_decibel_melgram=True, trainable_fb=False,
                       trainable_kernel=False,
                       name='mel_stft')
    m.trainable = False
    x = m(x)
    x = Normalization2D(int_axis=0, name='mel_stft_norm')(x)
    x = L.Permute((2, 1, 3))(x)
    x = L.Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim')(x)


    xFirst = L.Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = L.Dense(128)(xFirst)

    x = L.Dense(256, activation='relu')(query)
    x = L.Dense(128, activation='relu')(x)
   #x = L.Dense(64, activation='relu')(x)
   #x = L.Dense(32, activation='relu')(x)
   #x = L.Dense(16, activation='relu')(x)
    output = L.Dense(nCategories, activation='softmax', name='output')(x)
    model = Model(inputs=[inputs], outputs=[output])

    return model


def ConvModel(nCategories, samplingrate=16000,
                      inputLength=16000):
   # simple LSTM
    sr = samplingrate
    iLen = inputLength
    inputs = L.Input((inputLength,), name='input')

    x = L.Reshape((1, -1))(inputs)

    m = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, iLen),
                       padding='same', sr=sr, n_mels=80,
                       fmin=40.0, fmax=sr / 2, power_melgram=1.0,
                       return_decibel_melgram=True, trainable_fb=False,
                       trainable_kernel=False,
                       name='mel_stft')
    m.trainable = False
    x = m(x)

    x = Normalization2D(int_axis=0, name='mel_stft_norm')(x)
    x = L.Permute((2, 1, 3))(x)

    x = L.Conv2D(5, (5, 5), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(1, (5, 5), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)



    x = L.Lambda(lambda q: K.squeeze(q, -1))(x)
    xFirst = L.Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = L.Dense(64)(xFirst)

    x = L.Dense(32, activation='relu')(query)
    x = L.Dense(32, activation='relu')(x)
    output = L.Dense(nCategories, activation='softmax', name='output')(x)
    model = Model(inputs=[inputs], outputs=[output])

    return model



def RNNConvModel(nCategories, samplingrate=16000,
                      inputLength=16000):
   # simple LSTM
    sr = samplingrate
    iLen = inputLength
    inputs = L.Input((inputLength,), name='input')

    x = L.Reshape((1, -1))(inputs)

    m = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, iLen),
                       padding='same', sr=sr, n_mels=80,
                       fmin=40.0, fmax=sr / 2, power_melgram=1.0,
                       return_decibel_melgram=True, trainable_fb=False,
                       trainable_kernel=False,
                       name='mel_stft')
    m.trainable = False
    x = m(x)

    x = Normalization2D(int_axis=0, name='mel_stft_norm')(x)
    x = L.Permute((2, 1, 3))(x)

    x = L.Conv2D(1, (5, 5), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(1, (5, 5), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(1, (5, 5), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)

    x = L.Lambda(lambda q: K.squeeze(q, -1))(x)

    x = L.Bidirectional(L.LSTM(64, return_sequences=True))(x)
    x = L.Bidirectional(L.LSTM(32, return_sequences=True))(x) 

    y = L.Lambda(lambda q: q[:, -1])(x)  

    query = L.Dense(128)(y)

    x = L.Dense(64, activation='relu')(query)
    x = L.Dense(32, activation='relu')(x)
    output = L.Dense(nCategories, activation='softmax', name='output')(x)
    model = Model(inputs=[inputs], outputs=[output])

    return model

    


def AttentionRNNConvModel(nCategories, samplingrate=16000,
                      inputLength=16000):
    # simple LSTM
    sr = samplingrate
    iLen = inputLength
    inputs = L.Input((inputLength,), name='input')

    x = L.Reshape((1, -1))(inputs)

    m = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, iLen),
                       padding='same', sr=sr, n_mels=80,
                       fmin=40.0, fmax=sr / 2, power_melgram=1.0,
                       return_decibel_melgram=True, trainable_fb=False,
                       trainable_kernel=False,
                       name='mel_stft')
    m.trainable = False
    x = m(x)

    x = Normalization2D(int_axis=0, name='mel_stft_norm')(x)
    x = L.Permute((2, 1, 3))(x)

    x = L.Conv2D(1, (5, 5), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(1, (5, 5), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)

    x = L.Lambda(lambda q: K.squeeze(q, -1))(x)

    x = L.Bidirectional(L.LSTM(64, return_sequences=True))(x)
    x = L.Bidirectional(L.LSTM(32, return_sequences=True))(x) 

    y = L.Lambda(lambda q: q[:, -1])(x)  

    query = L.Dense(64)(y)
    attScores = L.Dot(axes=[1, 2])([query, x])
    attScores = L.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]
    attVector = L.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]
    
    x = L.Dense(64, activation='relu')(attVector)
    x = L.Dense(32, activation='relu')(x)
    output = L.Dense(nCategories, activation='softmax', name='output')(x)
    model = Model(inputs=[inputs], outputs=[output])

    return model