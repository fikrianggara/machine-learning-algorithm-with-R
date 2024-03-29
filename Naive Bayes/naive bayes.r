---
title: "naive bayes"
author: "Tim Modul"
date: "10/10/2020"
output: html_document
---

### Load Library
Tiga library yang dibutuhkan, yaitu **naivebayes, psych, dan caret**. Jika belum terinstall, silahkan install terlebih dahulu dengan perintah `install.packages("nama-package")`.

Library **naivebayes** akan digunakan untuk membuat modelnya. Library **psych** akan digunakan untuk melihat korelasi antar variabel. Library **caret** digunakan untuk membuat confusion matriks dan melihar akurasi model.

```{r message=FALSE, warning=FALSE}
library(naivebayes)
library(psych)
library(caret)
```


### Baca Data
Data tersimpan di folder `dataset`
```{r}
car <- read.csv("../dataset/car.txt", header=FALSE)
head(car)
```

Deskripsi data car bisa diliat di file car_info_var, V7 merupakan target class yaitu car acceptance

### Konversi Data
Ubah tipe variabel menjadi tipe faktor
```{r}
for(i in names(car)){
  car[,i]= as.factor(car[,i])
}
str(car)
```

### Pair Plot
Melihat korelasi dari tiap variabel, kalau ada korelasi yang tinggi, hilangkan salah satu variabel
```{r}
pairs.panels(car)
```

### Split Data
Memecah data menjadi data training(80% dari data awal) dan data test (20% dari data awal)
```{r}
set.seed(1234)
sampel <- sample(2, nrow(car), replace = T, prob = c(0.8,0.2))
trainingdat <- car[sampel==1, ]
testingdat <- car[sampel==2, ]
print(paste("Jumlah Train Data: ", nrow(trainingdat), "| Jumlah Test Data: ", nrow(testingdat)))
```

### Membuat Model
Gunakan atribut `laplace` untuk menghilangkan zero probability problem
```{r message=FALSE, warning=FALSE}
modelnaiv <- naive_bayes(V7~.,data=trainingdat)
modelnaiv
```
Summary Model
```{r}
summary(modelnaiv)
```

### Confusion Matrix
```{r}
prediksi <- predict(modelnaiv, testingdat)
confusionMatrix(table(prediksi,testingdat$V7))
```

