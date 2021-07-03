---
title: "Neural Network"
author: "Tim Modul 60"
date: "10/26/2020"
output: html_document
---
### Load Library
Tiga library yang dibutuhkan, yaitu **neuralnet dan caret**. Jika belum terinstall, silahkan install terlebih dahulu dengan perintah `install.packages("nama-package")`.

Library **neuralnet** akan digunakan untuk membuat model neural network. Library **caret** digunakan untuk membuat confusion matriks dan melihar akurasi model.

```{r message=FALSE, warning=FALSE}
library(caret)
library(neuralnet)

#keras dan tensorflow
install.packages("keras")
library(keras)
install.packages("tensorflow")
library(tensorflow)
install_tensorflow()

```

### Baca Data
Data tersimpan di folder `dataset`
```{r}
ipeh <- read.csv("../dataset/data_Ipeh.csv", header=T)
head(ipeh)
```


### Konversi dan normalisasi data
Ubah tipe variabel menjadi tipe faktor
```{r}
#nmelihat struktur variabel
str(ipeh)

```

```{r}
#normalisasi dengna feature scalink
normalisasi <- function(r){
  return((r-min(r))/(max(r)-min(r)))
}

# normalisasi semua atribut kecuali target class
for(i in colnames(ipeh[-1])){
    ipeh[ ,i]=normalisasi(ipeh[ ,i])
}

str(ipeh)
```

### Split Data
Memecah data menjadi data training (80% dari data awal) dan data test (20% dari data awal)
```{r}
set.seed(666)

ipehmat<-as.matrix(ipeh)
dimnames(ipehmat)<-NULL

sampel <- sample(2,nrow(ipehmat),replace = T, prob = c(0.8,0.2))
trainingdat <- ipehmat[sampel==1, ]
testingdat <- ipehmat[sampel==2, ]
print(paste("Jumlah train data :", nrow(trainingdat)))
print(paste("Jumlah test data :", nrow(testingdat)))

trainingdatfeature<-trainingdat[ ,2:4]
testingdatfeature<-testingdat[ ,2:4]
trainingdattarget<-trainingdat[,1]
testingdattarget<-testingdat[,1]

trainingdattarget<-to_categorical(trainingdattarget)
testingdattarget<-to_categorical(testingdattarget)
```
### Membuat Model
Misal kita ingin menggunakan semua atributnya
```{r}
set.seed(223)
#model dengan 1 hidden layer dan hidden node
modelnn<-neuralnet(admit~gre+gpa+rank, data=trainingdat,
                   hidden = 1,
                   err.fct = "ce",
                   linear.output = F)
plot(modelnn)
```

```{r}
#model dengan 1 hidden layer dan 5 hidden node
modelnn5<-neuralnet(admit~gre+gpa+rank, data=trainingdat,
                   hidden = 5,
                   err.fct = "ce",
                   linear.output = F)
plot(modelnn5)
```

```{r}
#model dengan 2 hidden layer, masing masing 2 hidden node dan 1 hidden node
modelnn21<-neuralnet(admit~gre+gpa+rank, data=trainingdat,
                   hidden = c(5,4),
                   err.fct = "ce",
                   linear.output = F)
plot(modelnn21)
```
**err.fct** merupakan loss function, fungsi yang digunakan untuk melihat seberapa besar error/loss yang dilakukan model dalam memprediksi, pilihan fungsi berupa sum square error **"sse"**, atau cross entropy **"ce"**.

**hidden** merupakan banyaknya hidden layer dan hidden node pada hidden layer yang akan dibuat. defaultnya, hanya terdapat satu hidden layer dan satu hidden node. jika ingin mengubah banyaknya hidden layer dan hidden node tiap layer, gunakan list (contoh hidden = c(5,4) , artinya terdapat dua hidden layer, hidden layer 1 mempunyai 5 hidden node, hidden layer 2 memiliki 4 hidden node).semakin banyak hidden node dan layer, komputasi yang dilakukan semakin mahal, namun bisa mengurangi error.

garis dan node biru merupakan bias dan penimbangnya.

**set.seed** diperlukan untuk menyimpan nilai penimbang yang random. jika tidak digunakan, penimbang yang digunakan akan terus berbeda beda setiap menjalankan perintah **neuralnet**.

fungsi aktivasi default adalah fungsi sigmoid, untuk mengubah fungsi aktivasi gunakan atribut **act.fct**. fungsi lain yang tersedia adalah fungsi tangent hyperbolic **"tanh"**.

baca atribut lain lebih lanjut dengan menjalankan **?neuralnet**

### Prediksi

jika output dari model lebih dari 0.5, maka kategorikan sebagai 1 (admitted), dan 0 (non admitted) jika lainnya

```{r}
# 1 hidden layer dan hidden node
prediksi <- compute(modelnn, testingdat[ ,-1])
pred <- ifelse(prediksi$net.result>0.5, 1, 0)
head(pred)
```

```{r}
#5 hidden node
prediksi5 <- compute(modelnn5, testingdat[ ,-1])
pred5 <- ifelse(prediksi5$net.result>0.5, 1, 0)
head(pred5)
```

```{r}
#2 hidden layer, 2 hidden node dan 1 hidden node
prediksi21 <- compute(modelnn21, testingdat[ ,-1])
pred21 <- ifelse(prediksi21$net.result>0.5, 1, 0)
head(pred21)
```

### Evaluasi Model

#### 1 hidden layer dan hidden node
```{r}
confusionMatrix(table(pred, testingdat$admit))
```

#### 5 hidden node
```{r}
confusionMatrix(table(pred5, testingdat$admit))
```

#### Hidden layer, 2 hidden node dan 1 hidden node
```{r}
confusionMatrix(table(pred21, testingdat$admit))
```

terlihat bahwa dari penambahan hidden layer dan hidden node tidak serta merta menaikan akurasi model.

maap ya ges neuralnya baru dibikin, baru ketemu sumbernya soale ehheuheuhe. maap juga kalo dokumentasinya terlalu panjang dan malah bikin bingung.


### deep Neural Network Menggunakan Keras
install terlebih dahulu package keras dan tensorflow, nanti di package keras diminta download miniconda, total penyimpanan yang dibutuhkan kira kira sekitar 2.5 GB
```{r}
#membuat model
modelkeras<-keras_model_sequential()
modelkeras %>% 
    layer_dense(units = 8, activation = 'relu', input_shape = c(3)) %>% 
    layer_dense(units = 2, activation = 'softmax')
summary(modelkeras)
```
units pada layer_dense pertama berarti bahwa hidden layer pertama memiliki 8 node, activatioin merupakan fungsi aktivasi yang digunakan, pada kasus ini digunakan relu, input_shape merupakan banyaknya kolom pada data training. layer_dense yang kedua merupakan layer yang kedua (karena hanya menggunakan dua layer, layer_dense yang terkhir merupakan output layer). units pada layer terakhir merupakan banyaknya kelas yang akan diprediksi, karena kita memprediksi admit atau tidak admit, maka units nya bernilai 2.

#### kompilasi
```{r}
modelkeras %>% compile(
     loss = 'binary_crossentropy',
     optimizer = 'adam',
     metrics = 'accuracy'
 )
```
loss merupakan fungsi los, digunakan binary cross entropy karena kelas yang digunakan hanya dua (categorical_crossentropy untuk ), optimizer merupakan metode yang digunakan untuk mengoptimasi loss (seperti gradient descent), metrics, merupakan pengukuran yang akan ditampilkan saat fitting.

