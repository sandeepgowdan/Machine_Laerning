## Caret##
install.packages("caret")
library(caret)

## Explore the data
head(iris)
str(iris)


## visulise the data
featurePlot(x=iris[,1:4],y=iris$Species, plot="ellipse")

featurePlot(x=iris[,1:4],y=iris$Species,plot="ellipse",auto.key=list(columns=3))

featurePlot(x=iris[,1:4],y=iris$Species,plot=c("boxplot"),auto.key=list(columns=3),
            scale=list(y=list(relation="free")),labels=c("Species","length in cm"))

featurePlot(x=iris[,1:4],y=iris$Species,plot="density",auto.key=list(columns=3),
            labels=c("Species",""),scale=list(x=list(relation="free"),y=list(relation="free")),pch="|")

featurePlot(x=iris[,1:3],y=iris[,4],plot="scatter",type=c("p","smooth"),
            auto.key=list(columns=3),pch=16,lwd=2)

###See correlation
library(corrplot)

cor_iris=cor(iris[,1:4])

corrplot(cor_iris, type="upper",method="ellipse")


### Cross validation
trc= trainControl(method="repeatedcv",number=10,repeats=3)
trc

#Fit model 
mod_lm = train(Sepal.Length~.,iris,method="lm",trControl=trc)
mod_lm
summary(mod_lm)

mod_lm$resample
mod_lm$results

#####   diagnostic plots  #########
plot(mod_lm$finalModel)


####  cstom plots to test fit
obs=predict(mod_lm, iris)

plot(iris$Sepal.Length,obs)
abline(0,1)

plot(1:nrow(iris),iris$Sepal.Length)
points(1:nrow(iris),obs, col="red")

###  prediction from new data
df_new=data.frame(Sepal.Width=c(4.8,4.6), Petal.Length=c(3.4,6.9), 
                  Petal.Width=c(1.7,2.4), Species=c("setosa","virginica"))
predict(mod_lm,df_new)


######   other models using caret : classification models  ######
#support vector machine
mod_svm= train(Species~.,iris,method="svmRadial",trControl=trc)
mod_svm
obs=predict(mod_svm, iris)

confusionMatrix(iris$Species,obs)
plot(mod_svm)

#Gradient Boosting Machine
mod_gbm= train(Species~.,iris,method="gbm",trControl=trc)
mod_gbm
obs=predict(mod_gbm, iris)

confusionMatrix(iris$Species,obs)
plot(mod_gbm)


#Learning Vector Quantization  model
mod_lvq= train(Species~.,iris,method="lvq",trControl=trc)
mod_lvq
obs=predict(mod_lvq, iris)

confusionMatrix(iris$Species,obs)
plot(mod_lvq)

#Random Forest  model
mod_rf= train(Species~.,iris,method="rf",trControl=trc)
mod_rf
obs=predict(mod_rf, iris)

confusionMatrix(iris$Species,obs)
plot(mod_rf)
