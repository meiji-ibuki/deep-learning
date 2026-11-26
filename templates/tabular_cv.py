import numpy as np
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from pathlib import Path
X, y = make_classification(n_samples=1200, n_features=16, random_state=42)
X, test, y, yt = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
oof = np.full(len(y), np.nan)
preds = []
for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X,y)):
    a,b = train_test_split(tr, test_size=.2, stratify=y[tr], random_state=fold)
    prep = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    xa = prep.fit_transform(X[a]).astype('float32')
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42+fold)
    model = tf.keras.Sequential([tf.keras.Input((X.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
    model.fit(xa,y[a],validation_data=(prep.transform(X[b]),y[b]),epochs=30,batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=4,restore_best_weights=True)],verbose=0)
    oof[va] = model.predict(prep.transform(X[va]),verbose=0).ravel()
    preds.append(model.predict(prep.transform(test),verbose=0).ravel())
assert np.isfinite(oof).all()
p = np.mean(preds,axis=0)
print('OOF AUC',roc_auc_score(y,oof),'test AUC',roc_auc_score(yt,p))
Path('outputs').mkdir(exist_ok=True)
np.savez('outputs/tabular.npz',oof=oof,test=p,y=y,yt=yt)
