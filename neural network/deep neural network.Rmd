---
title: "Deep Neural Network"
author: "Fikri Septrian Anggara"
date: "1/10/2021"
output: html_document
---

### load library keras dan tensorflow
install terlebih dahulu package tensorflow dan keras, kemudian masukan kode install_tensorflow dan install_keras

dibutuhkan waktu yang lumayan lama dalam penginstallan, penyimpanan yang dibutuhkan juga lumayan besar, nanti akan diminta menginstall dependensi seperti miniconda dll. ikuti saja, dan baca errornya/copas ke google jika mengalami kendala.
```{r}
#keras dan tensorflow
library(keras)
library(tensorflow)
library(tidyverse)

```

### load data
```{r}
ipeh <- read.csv("../dataset/data_Ipeh.csv", header=T)
head(ipeh)


#normalisasi dengna feature scalink
normalisasi <- function(r){
  return((r-min(r))/(max(r)-min(r)))
}

# normalisasi semua atribut kecuali target class
for(i in colnames(ipeh[-1])){
    ipeh[ ,i]=normalisasi(ipeh[ ,i])
}

str(ipeh)

ipeh %>% ggplot(aes(x=as.factor(admit)))+geom_bar()
ipeh %>% group_by(admit) %>% summarise(banyak=n())

#karena kelas tidak balance, maka kelas 0=273 diambil sebanyak kelas 1=127. sehingga tiap kelas sama sama memiliki 127 observasi.

ipehadmit0<-ipeh %>% filter(admit==0)
ipehadmit1<-ipeh %>% filter(admit==1)

sample<-sample(273,127,F)
ipehadmit0fix<-ipehadmit0[sample, ]

ipehfix<-rbind(ipehadmit0fix,ipehadmit1)

ipehfix %>% ggplot(aes(x=as.factor(admit)))+geom_bar()
#kelas sudah balance
```


### split data
```{r}
set.seed(666)

ipehmat<-as.matrix(ipehfix)
dimnames(ipehmat)<-NULL

sampel <- sample(2,nrow(ipehmat),replace = T, prob = c(0.8,0.2))
trainingdat <- ipehmat[sampel==1, ]
testingdat <- ipehmat[sampel==2, ]
print(paste("Jumlah train data :", nrow(trainingdat)))
print(paste("Jumlah test data :", nrow(testingdat)))

trainingdatfeature<-trainingdat[ ,2:4]
testingdatfeature<-testingdat[ ,2:4]
trainingdattarget<-to_categorical(trainingdat[,1])
testingdattarget<-to_categorical(testingdat[,1])
```

### membuat model deep neural network dengan 3 hidden layer
```{r}
#membuat model
modelkeras<-keras_model_sequential()
modelkeras %>% 
    layer_dense(units = 20, activation = 'relu', input_shape = c(3)) %>% 
    layer_dense(units = 30, activation = 'relu') %>% 
    layer_dense(units = 30, activation = 'relu') %>% 
    layer_dense(units = 2, activation = 'sigmoid')

summary(modelkeras)
```
units pada layer_dense pertama berarti bahwa hidden layer pertama memiliki 20 node, activation merupakan fungsi aktivasi yang digunakan, pada kasus ini digunakan relu, input_shape merupakan banyaknya kolom pada data training. layer_dense yang kedua merupakan hidden layer kedua, layer_Dense ketiga merupakan hidden layer ketiga, dst, layer_dense terakhir merupakan layer output, units pada layer ini menggambarkan banyaknya kelas yang akan diprediksi, karena kita menggunakan admit dan tidak admit (2 kelas), maka units yang digunakan adalah 2. sigmoid digunakan untuk karena memiliki nilai 0-1, yang digunakan sebagai estimasi peluang masuk ke kelas tertentu.

### kompilasi
```{r}
modelkeras %>% compile(
     loss = 'binary_crossentropy',
     optimizer = 'adam',
     metrics = 'accuracy'
 )
```
loss merupakan fungsi los, digunakan binary cross entropy karena kelas yang digunakan hanya dua (categorical_crossentropy untuk ), optimizer merupakan metode yang digunakan untuk mengoptimasi loss (seperti gradient descent), metrics, merupakan pengukuran yang akan ditampilkan saat training model.

### Fitting
```{r}
history<-modelkeras %>% fit(
     trainingdatfeature, 
     trainingdattarget, 
     epochs =300, 
     batch_size = 20, 
     validation_split = 0.2
 )
plot(history)
```
epoch merupakan banyaknya pengulangan model melakukan training, semakin banyak perulangan semakin banyak waktu yang diperlukan, semakin tinggi akurasi yang dihasilkan. batch size merupakan banyaknya data yang digunakan dalam sekali epoch. validation split merupakan proporsi data yang digunakan untuk validasi.

### plot proses training

#### loss
```{r}
# Plot loss dari data training
plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l")

# Plot loss dari data testing
lines(history$metrics$val_loss, col="red")

#legend
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))
```
#### akurasi
```{r}
# Plot akurasi dari data training
plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l")

# Plot akurasi dari data validasi
lines(history$metrics$val_acc, col="green")

# Add Legend
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))
```

### testing
```{r}
set.seed(123)
classes <- modelkeras %>% predict_classes(testingdatfeature, batch_size = 20)
```

```{r}
table(testingdattarget[,2],classes)
```


