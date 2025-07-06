# Kasifikasi_Menggunakan_metode_CNN

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==== 1. Data minimal 20 ====
data = {
    'berita': [
        "Presiden resmikan gedung baru di Jakarta",
        "Minyak kayu putih bisa obati kanker",
        "Gempa bumi melanda wilayah timur Indonesia",
        "Orang bisa hidup tanpa makan selama 1 tahun",
        "Pemerintah luncurkan program beasiswa baru",
        "Hujan uang terjadi di jalanan kota",
        "BMKG umumkan potensi tsunami di Aceh",
        "Bumi itu datar menurut penelitian",
        "Peluncuran satelit berhasil dilakukan pagi ini",
        "Ilmuwan temukan obat corona dari daun jambu",
        "Kebakaran hutan di Kalimantan berhasil dipadamkan",
        "Garam bisa sembuhkan semua penyakit",
        "Menteri umumkan kenaikan gaji PNS tahun depan",
        "Matahari terbit dari barat mulai minggu depan",
        "Atlet Indonesia raih emas di olimpiade",
        "Roti bisa menyebabkan kanker otak",
        "Bandara baru mulai beroperasi bulan ini",
        "Cacing bisa hidup di otak manusia",
        "Presiden tandatangani undang-undang baru",
        "Konsumsi air es bikin otak beku"
    ],
    'label': [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]  # 0=Asli, 1=Hoaks
}
df = pd.DataFrame(data)

# ==== 2. Tokenisasi ====
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['berita'])
sequences = tokenizer.texts_to_sequences(df['berita'])
padded = pad_sequences(sequences, maxlen=10, padding='post')

# ==== 3. Split Data ====
X_train, X_test, y_train, y_test = train_test_split(padded, df['label'], test_size=0.2, random_state=42)

# ==== 4. Model CNN ====
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=10),
    Conv1D(32, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)

# ==== 5. Prediksi ====
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# ==== 6. Grafik Akurasi dan Loss ====
plt.figure(figsize=(10,7))
plt.title('Grafik Akurasi dan Loss Selama Pelatihan CNN', fontsize=14)
epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['accuracy'], 'g-o', label='Train Accuracy')
plt.plot(epochs, history.history['val_accuracy'], 'orange', marker='x', linestyle='--', label='Validation Accuracy')
plt.plot(epochs, history.history['loss'], 'b-s', label='Train Loss')
plt.plot(epochs, history.history['val_loss'], 'red', marker='^', linestyle=':', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Nilai')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==== 7. Confusion Matrix ====
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Asli', 'Hoaks'], yticklabels=['Asli', 'Hoaks'])
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# ==== 8. Classification Report ====
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Asli', 'Hoaks']))

# ==== 9. Prediksi Beberapa Berita Baru ====
berita_baru = [
    "Menteri umumkan penambahan subsidi listrik",
    "Air kelapa bisa menyembuhkan tumor otak"
]
seq_baru = tokenizer.texts_to_sequences(berita_baru)
pad_baru = pad_sequences(seq_baru, maxlen=10, padding='post')
pred_baru = model.predict(pad_baru)

for i, kalimat in enumerate(berita_baru):
    label = "Hoaks" if pred_baru[i][0] > 0.5 else "Asli"
    print(f"Berita: {kalimat}")
    print(f"Prediksi: {label} (Probabilitas: {pred_baru[i][0]:.2f})\n")

    
