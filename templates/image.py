import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers as L
tf.keras.utils.set_random_seed(42)
(x,y),(xt,yt)=tf.keras.datasets.fashion_mnist.load_data()
a,v,b,w=train_test_split(x,y,test_size=.15,stratify=y,random_state=42)
def ds(x,y,train=False):
    d=tf.data.Dataset.from_tensor_slices((x[...,None],y))
    if train: d=d.shuffle(len(y),seed=42)
    return d.batch(128).prefetch(tf.data.AUTOTUNE)
m=tf.keras.Sequential([L.Input((28,28,1)),L.Rescaling(1/255),L.RandomTranslation(.08,.08),L.Conv2D(32,3,activation='relu'),L.MaxPooling2D(),L.Conv2D(64,3,activation='relu'),L.GlobalAveragePooling2D(),L.Dense(10,activation='softmax')])
m.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
m.fit(ds(a,b,True),validation_data=ds(v,w),epochs=15,callbacks=[tf.keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)])
print(m.evaluate(ds(xt,yt)))
m.save('image.keras')
