library(readr)
library(tm)
library(class)
library(plyr)
library(klaR) #Classification Lib
library(randomForest)
library(RWeka)

job.desc <-read_tsv(file.choose())  # Train.tsv
#job.desc.test <- read_tsv(file.choose())


str(job.desc)
#str(job.desc.test)

summary(job.desc)
#summary(job.desc.test)

job.desc <- as.data.frame(job.desc)
#job.desc.test <- as.data.frame(job.desc.test)


job.desc$tags <- as.character(job.desc$tags)


tags.all <- c("part-time-job","full-time-job","hourly-wage","salary","associate-needed","bs-degree-needed","ms-or-phd-needed","licence-needed","1-year-experience-needed","2-4-years-experience-needed","5-plus-years-experience-needed","supervising-job")

sapply(job.desc,function(x){sum(is.na(x))})


job.desc <- job.desc[-which(is.na(job.desc$tags)),]
str(job.desc)

# Creating different columns for different tags
tmp <- strsplit(job.desc$tags, " ")  # splits row wise

job.desc[,tags.all] <- 0
#job.desc.test[,tags.all] <- 0

# Adding 1 to the descriptions having the tags
for(i in 1:nrow(job.desc)){
  for(j in 1:length(tmp[[i]])){
    column <- tmp[[i]][j]
    job.desc[i,column] <- 1
  }
  
}

str(job.desc)
#str(job.desc.test)


#job.desc.test$tags <- ""

#job.desc.stack <- rbind(job.desc,job.desc.test)

#str(job.desc.stack)

# Creating corpus for training data

job.cor <- Corpus(VectorSource(job.desc$description))
#job.cor <- Corpus(VectorSource(job.desc.stack$description))

cleanCorpus <- function(corpus){
  corpus.tmp <- tm_map(corpus, removePunctuation)
  corpus.tmp <- tm_map(corpus.tmp, stripWhitespace)
  corpus.tmp <- tm_map(corpus.tmp, tolower)
  corpus.tmp <- tm_map(corpus.tmp, removeWords,stopwords("english"))
  return(corpus.tmp)
}


job.cor.cl <- cleanCorpus(job.cor)
job.tdm <- TermDocumentMatrix(job.cor.cl)
str(job.tdm)
names(job.tdm)
job.tdm <- removeSparseTerms(job.tdm,0.92)
str(job.tdm)
#result <- list(name = cand, tdm = job.tdm)


job.mat <- t(data.matrix(job.tdm)) # transpoing, 
job.df <- as.data.frame(job.mat)

names(job.df)
colnames(job.df)[which(names(job.df) == "salary")] <- "salaryInText"

job.df <- cbind(job.df, job.desc[,c(3:14)])
#job.df <- cbind(job.df, job.desc.stack[,c(3:14)])

# Modeling

 train.idx <- sample(nrow(job.df),ceiling(nrow(job.df)*0.7))
 test.idx <- (1:nrow(job.df))[-train.idx]
# head(test.idx)

# train.idx <- 1:3504
# test.idx  <- 3505:6425
##############################################################################################################
# Model Naive Bayes

job.nl <- job.df[,!colnames(job.df) %in% tags.all] # selcting only the dependent variables


job.nl.partTimeJob               <- job.nl
job.nl.fullTimeJob               <- job.nl
job.nl.hourlyWage                <- job.nl
job.nl.salary                    <- job.nl
job.nl.associateNeeded           <- job.nl
job.nl.bsNeeded                  <- job.nl
job.nl.msPhdNeeded               <- job.nl
job.nl.licenceNeeded             <- job.nl
job.nl.1yearExpNeeded            <- job.nl
job.nl.2yearExpNeeded            <- job.nl
job.nl.5yearExpNeeded            <- job.nl
job.nl.supervisingJob            <- job.nl


job.nl.partTimeJob[,"part-time-job"]                <- job.df[,"part-time-job"]
job.nl.fullTimeJob[,"full-time-job"]                <- job.df[,"full-time-job"]
job.nl.hourlyWage[,"hourly-wage"]                  <- job.df[,"hourly-wage"]
job.nl.salary[,"salary"]                       <- job.df[,"salary"]
job.nl.associateNeeded[,"associate-needed"]             <- job.df[,"associate-needed"]
job.nl.bsNeeded[,"bs-degree-needed"]             <- job.df[,"bs-degree-needed"]
job.nl.msPhdNeeded[,"ms-or-phd-needed"]             <- job.df[,"ms-or-phd-needed"]
job.nl.licenceNeeded[,"licence-needed"]               <- job.df[,"licence-needed"]
job.nl.1yearExpNeeded[,"1-year-experience-needed"]     <- job.df[,"1-year-experience-needed"]
job.nl.2yearExpNeeded[,"2-4-years-experience-needed"]  <- job.df[,"2-4-years-experience-needed"]
job.nl.5yearExpNeeded[,"5-plus-years-experience-needed"]    <- job.df[,"5-plus-years-experience-needed"]
job.nl.supervisingJob[,"supervising-job"]              <- job.df[,"supervising-job"]


job.nl.partTimeJob[,"part-time-job"]         <-			as.factor(job.nl.partTimeJob[,"part-time-job"])                    	
job.nl.fullTimeJob[,"full-time-job"]         <-     		as.factor(job.nl.fullTimeJob[,"full-time-job"])                    
job.nl.hourlyWage[,"hourly-wage"]            <-     		as.factor(job.nl.hourlyWage[,"hourly-wage"])                       
job.nl.salary[,"salary"]       <-     				as.factor(job.nl.salary[,"salary"])                                
job.nl.associateNeeded[,"associate-needed"]         <-     	as.factor(job.nl.associateNeeded[,"associate-needed"])             
job.nl.bsNeeded[,"bs-degree-needed"]         <-     		as.factor(job.nl.bsNeeded[,"bs-degree-needed"])                    
job.nl.msPhdNeeded[,"ms-or-phd-needed"]      <-     		as.factor(job.nl.msPhdNeeded[,"ms-or-phd-needed"])                 
job.nl.licenceNeeded[,"licence-needed"]      <-     		as.factor(job.nl.licenceNeeded[,"licence-needed"])                 
job.nl.1yearExpNeeded[,"1-year-experience-needed"]         <-   as.factor(job.nl.1yearExpNeeded[,"1-year-experience-needed"])        
job.nl.2yearExpNeeded[,"2-4-years-experience-needed"]      <-   as.factor(job.nl.2yearExpNeeded[,"2-4-years-experience-needed"])     
job.nl.5yearExpNeeded[,"5-plus-years-experience-needed"]   <-   as.factor(job.nl.5yearExpNeeded[,"5-plus-years-experience-needed"])  
job.nl.supervisingJob[,"supervising-job"]           <-     	as.factor(job.nl.supervisingJob[,"supervising-job"])               


job.nb.partTimeJob <- AdaBoostM1(job.nl.partTimeJob[train.idx,]$`part-time-job` ~ ., data = job.nl.partTimeJob[train.idx,],control = Weka_control(W = list(J48, C=.5, M = 10)))
job.nb.pred.partTimeJob <- predict(job.nb.partTimeJob, job.nl.partTimeJob[test.idx,])



job.nb.fullTimeJob <- AdaBoostM1(job.nl.fullTimeJob[train.idx,]$`full-time-job` ~ ., data = job.nl.fullTimeJob[train.idx,],control = Weka_control(W = list(J48, C=.5, M = 10)))
job.nb.pred.fullTimeJob <- predict(job.nb.fullTimeJob, job.nl.fullTimeJob[test.idx,])


job.nb.hourlyWage <- AdaBoostM1(job.nl.hourlyWage[train.idx,]$`hourly-wage` ~ ., data = job.nl.hourlyWage[train.idx,],control = Weka_control(W = list(J48, C=.5, M = 10)))
job.nb.pred.hourlyWage <- predict(job.nb.hourlyWage, job.nl.hourlyWage[test.idx,])


job.nb.salary <- AdaBoostM1(job.nl.salary[train.idx,]$salary ~ ., data = job.nl.salary[train.idx,],control = Weka_control(W = list(J48, C=.5, M = 10)))
job.nb.pred.salary <- predict(job.nb.salary, job.nl.salary[test.idx,])


job.nb.associateNeeded <- AdaBoostM1(job.nl.associateNeeded[train.idx,]$`associate-needed` ~ ., data = job.nl.associateNeeded[train.idx,],control = Weka_control(W = list(J48, C=.5, M = 10)))
job.nb.pred.associateNeeded <- predict(job.nb.associateNeeded, job.nl.associateNeeded[test.idx,])


job.nb.bsNeeded <- AdaBoostM1(job.nl.bsNeeded[train.idx,]$`bs-degree-needed` ~ ., data = job.nl.bsNeeded[train.idx,],control = Weka_control(W = list(J48, C=.5, M = 10)))
job.nb.pred.bsNeeded <- predict(job.nb.bsNeeded, job.nl.bsNeeded[test.idx,])


job.nb.msPhdNeeded <- AdaBoostM1(job.nl.msPhdNeeded[train.idx,]$`ms-or-phd-needed` ~ ., data = job.nl.msPhdNeeded[train.idx,],control = Weka_control(W = list(J48, C=.5, M = 10)))
job.nb.pred.msPhdNeeded <- predict(job.nb.msPhdNeeded, job.nl.msPhdNeeded[test.idx,])


job.nb.licenceNeeded <- AdaBoostM1(job.nl.licenceNeeded[train.idx,]$`licence-needed` ~ ., data = job.nl.licenceNeeded[train.idx,],control = Weka_control(W = list(J48, C=.5, M = 10)))
job.nb.pred.licenceNeeded <- predict(job.nb.licenceNeeded, job.nl.licenceNeeded[test.idx,])


job.nb.1yearExpNeeded <- AdaBoostM1(job.nl.1yearExpNeeded[train.idx,]$`1-year-experience-needed` ~ ., data = job.nl.1yearExpNeeded[train.idx,],control = Weka_control(W = list(J48, C=.5, M = 10)))
job.nb.pred.1yearExpNeeded <- predict(job.nb.1yearExpNeeded, job.nl.1yearExpNeeded[test.idx,])


job.nb.2yearExpNeeded <- AdaBoostM1(job.nl.2yearExpNeeded[train.idx,]$`2-4-years-experience-needed` ~ ., data = job.nl.2yearExpNeeded[train.idx,],control = Weka_control(W = list(J48, C=.5, M = 10)))
job.nb.pred.2yearExpNeeded <- predict(job.nb.2yearExpNeeded, job.nl.2yearExpNeeded[test.idx,])


job.nb.5yearExpNeeded <- AdaBoostM1(job.nl.5yearExpNeeded[train.idx,]$`5-plus-years-experience-needed` ~ ., data = job.nl.5yearExpNeeded[train.idx,],control = Weka_control(W = list(J48, C=.5, M = 10)))
job.nb.pred.5yearExpNeeded <- predict(job.nb.5yearExpNeeded, job.nl.5yearExpNeeded[test.idx,])


job.nb.supervisingJob <- AdaBoostM1(job.nl.supervisingJob[train.idx,]$`supervising-job` ~ ., data = job.nl.supervisingJob[train.idx,],control = Weka_control(W = list(J48, C=.5, M = 10)))
job.nb.pred.supervisingJob <- predict(job.nb.supervisingJob, job.nl.supervisingJob[test.idx,])



conf.matrix.partTimeJob <- table("Predictions" = job.nb.pred.partTimeJob , Actual = job.nl.partTimeJob[test.idx,]$`part-time-job`)
accuracy.partTimeJob <- sum(diag(conf.matrix.partTimeJob))/length(test.idx)*100
accuracy.partTimeJob

conf.matrix.fullTimeJob <- table("Predictions" = job.nb.pred.fullTimeJob , Actual = job.nl.fullTimeJob[test.idx,]$`full-time-job`)
accuracy.fullTimeJob <- sum(diag(conf.matrix.fullTimeJob))/length(test.idx)*100
accuracy.fullTimeJob

conf.matrix.hourlyWage <- table("Predictions" = job.nb.pred.hourlyWage , Actual = job.nl.hourlyWage[test.idx,]$`hourly-wage`)
accuracy.hourlyWage <- sum(diag(conf.matrix.hourlyWage))/length(test.idx)*100
accuracy.hourlyWage

conf.matrix.salary <- table("Predictions" = job.nb.pred.salary , Actual = job.nl.salary[test.idx,]$salary)
accuracy.salary <- sum(diag(conf.matrix.salary))/length(test.idx)*100
accuracy.salary

conf.matrix.associateNeeded <- table("Predictions" = job.nb.pred.associateNeeded , Actual = job.nl.associateNeeded[test.idx,]$`associate-needed`)
accuracy.associateNeeded <- sum(diag(conf.matrix.associateNeeded))/length(test.idx)*100
accuracy.associateNeeded

conf.matrix.bsNeeded <- table("Predictions" = job.nb.pred.bsNeeded , Actual = job.nl.bsNeeded[test.idx,]$`bs-degree-needed`)
accuracy.bsNeeded <- sum(diag(conf.matrix.bsNeeded))/length(test.idx)*100
accuracy.bsNeeded

conf.matrix.msPhdNeeded <- table("Predictions" = job.nb.pred.msPhdNeeded , Actual = job.nl.msPhdNeeded[test.idx,]$`ms-or-phd-needed`)
accuracy.msPhdNeeded <- sum(diag(conf.matrix.msPhdNeeded))/length(test.idx)*100
accuracy.msPhdNeeded

conf.matrix.licenceNeeded <- table("Predictions" = job.nb.pred.licenceNeeded , Actual = job.nl.licenceNeeded[test.idx,]$`licence-needed`)
accuracy.licenceNeeded <- sum(diag(conf.matrix.licenceNeeded))/length(test.idx)*100
accuracy.licenceNeeded

conf.matrix.1yearExpNeeded <- table("Predictions" = job.nb.pred.1yearExpNeeded , Actual = job.nl.1yearExpNeeded[test.idx,]$`1-year-experience-needed`)
accuracy.1yearExpNeeded <- sum(diag(conf.matrix.1yearExpNeeded))/length(test.idx)*100
accuracy.1yearExpNeeded

conf.matrix.2yearExpNeeded <- table("Predictions" = job.nb.pred.2yearExpNeeded , Actual = job.nl.2yearExpNeeded[test.idx,]$`2-4-years-experience-needed`)
accuracy.2yearExpNeeded <- sum(diag(conf.matrix.2yearExpNeeded))/length(test.idx)*100
accuracy.2yearExpNeeded

conf.matrix.5yearExpNeeded  <- table("Predictions" = job.nb.pred.5yearExpNeeded  , Actual = job.nl.5yearExpNeeded [test.idx,]$`5-plus-years-experience-needed`)
accuracy.5yearExpNeeded  <- sum(diag(conf.matrix.5yearExpNeeded ))/length(test.idx)*100
accuracy.5yearExpNeeded

conf.matrix.supervisingJob  <- table("Predictions" = job.nb.pred.supervisingJob  , Actual = job.nl.supervisingJob [test.idx,]$`supervising-job`)
accuracy.supervisingJob  <- sum(diag(conf.matrix.supervisingJob ))/length(test.idx)*100
accuracy.supervisingJob
################  

# job.desc.test[,"part-time-job"]			 <-  job.nb.pred.partTimeJob        
# job.desc.test[,"full-time-job"]			 <-  job.nb.pred.fullTimeJob        
# job.desc.test[,"hourly-wage"]			 <-  job.nb.pred.hourlyWage         
# job.desc.test[,"salary"]			 <-  job.nb.pred.salary             
# job.desc.test[,"associate-needed"]		 <-  job.nb.pred.associateNeeded    
# job.desc.test[,"bs-degree-needed"]		 <-  job.nb.pred.bsNeeded           
# job.desc.test[,"ms-or-phd-needed"]		 <-  job.nb.pred.msPhdNeeded        
# job.desc.test[,"licence-needed"]		 <-  job.nb.pred.licenceNeeded      
# job.desc.test[,"1-year-experience-needed"]	 <-  job.nb.pred.1yearExpNeeded     
# job.desc.test[,"2-4-years-experience-needed"]	 <-  job.nb.pred.2yearExpNeeded     
# job.desc.test[,"5-plus-years-experience-needed"] <-  job.nb.pred.5yearExpNeeded     
# job.desc.test[,"supervising-job"]		 <-  job.nb.pred.supervisingJob     
# 
# 

########################################################   Model Random Forest    ########################################################

# 
# job.rf.partTimeJob <- randomForest(part-time-job ~ .,job.nl.partTimeJob[train.idx,],n.trees=100)
# View(job.nl.partTimeJob[train.idx,]$`part-time-job`)
# head(job.nl.partTimeJob[train.idx,]$`part-time-job`)
# 
# job.rf.pred.partTimeJob <- predict(job.rf.partTimeJob, job.nl.partTimeJob[test.idx,])





##############################################################################################################
# Model KNN   


# job.nl <- job.df[,!colnames(job.df) %in% tags.all] # selcting only the dependent variables
# head(job.nl)
# 
# job.dependent.partTimeJob <- job.df[,"part-time-job"]
# job.dependent.fullTimeJob <- job.df[,"full-time-job"]
# job.dependent.hourlyWage <- job.df[,"hourly-wage"]
# job.dependent.salary <- job.df[,"salary"]
# job.dependent.associateNeeded <- job.df[,"associate-needed"]
# job.dependent.bsNeeded <- job.df[,"bs-degree-needed"]
# job.dependent.msPhdNeeded <- job.df[,"ms-or-phd-needed"]
# job.dependent.licenceNeeded <- job.df[,"licence-needed"]
# job.dependent.1yearExpNeeded <- job.df[,"1-year-experience-needed"]
# job.dependent.2yearExpNeeded <- job.df[,"2-4-years-experience-needed"]
# job.dependent.5yearExpNeeded <- job.df[,"5-plus-years-experience-needed"]
# job.dependent.supervisingJob <- job.df[,"supervising-job"]
# 
# 
# knn.pred.partTimeJob <- knn(job.nl[train.idx,], job.nl[test.idx,], job.dependent.partTimeJob[train.idx])
# knn.pred.fullTimeJob <- knn(job.nl[train.idx,], job.nl[test.idx,], job.dependent.fullTimeJob[train.idx])
# knn.pred.hourlyWage <- knn(job.nl[train.idx,], job.nl[test.idx,], job.dependent.hourlyWage[train.idx])
# knn.pred.salary <- knn(job.nl[train.idx,], job.nl[test.idx,], job.dependent.salary[train.idx])
# knn.pred.associateNeeded <- knn(job.nl[train.idx,], job.nl[test.idx,], job.dependent.associateNeeded[train.idx])
# knn.pred.bsNeeded <- knn(job.nl[train.idx,], job.nl[test.idx,], job.dependent.bsNeeded[train.idx])
# knn.pred.msPhdNeeded <- knn(job.nl[train.idx,], job.nl[test.idx,], job.dependent.msPhdNeeded[train.idx])
# knn.pred.licenceNeeded <- knn(job.nl[train.idx,], job.nl[test.idx,], job.dependent.licenceNeeded[train.idx])
# knn.pred.1yearExpNeeded <- knn(job.nl[train.idx,], job.nl[test.idx,], job.dependent.1yearExpNeeded[train.idx])
# knn.pred.2yearExpNeeded <- knn(job.nl[train.idx,], job.nl[test.idx,], job.dependent.2yearExpNeeded[train.idx])
# knn.pred.5yearExpNeeded <- knn(job.nl[train.idx,], job.nl[test.idx,], job.dependent.5yearExpNeeded[train.idx])
# knn.pred.supervisingJob <- knn(job.nl[train.idx,], job.nl[test.idx,], job.dependent.supervisingJob[train.idx])
# 
# # Accuracy
# 
# conf.matrix <- table("Predictions" = knn.pred.partTimeJob , Actual = job.dependent.partTimeJob[test.idx])
# accuracy <- sum(diag(conf.matrix))/length(test.idx)*100
# 
# 
# job.desc.test[,"part-time-job"]			 <-  knn.pred.partTimeJob        
# job.desc.test[,"full-time-job"]			 <-  knn.pred.fullTimeJob        
# job.desc.test[,"hourly-wage"]			 <-  knn.pred.hourlyWage         
# job.desc.test[,"salary"]			 <-  knn.pred.salary             
# job.desc.test[,"associate-needed"]		 <-  knn.pred.associateNeeded    
# job.desc.test[,"bs-degree-needed"]		 <-  knn.pred.bsNeeded           
# job.desc.test[,"ms-or-phd-needed"]		 <-  knn.pred.msPhdNeeded        
# job.desc.test[,"licence-needed"]		 <-  knn.pred.licenceNeeded      
# job.desc.test[,"1-year-experience-needed"]	 <-  knn.pred.1yearExpNeeded     
# job.desc.test[,"2-4-years-experience-needed"]	 <-  knn.pred.2yearExpNeeded     
# job.desc.test[,"5-plus-years-experience-needed"] <-  knn.pred.5yearExpNeeded     
# job.desc.test[,"supervising-job"]		 <-  knn.pred.supervisingJob     
# 
# 
# 
# for(i in 1:nrow(job.desc.test)){
#   tags <- NULL
#   for(j in tags.all){
#     if(job.desc.test[i,j] == 1){
#       tags <- c(tags,j)
#     }
#   }
#   job.desc.test[i,"tags"] <-  paste(tags,collapse = " ")
# }
# write_tsv(data.frame(job.desc.test$tags),"tags.tsv")
