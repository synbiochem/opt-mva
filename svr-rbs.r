# RBS-ML, Pablo Carbonell SYNBIOCHEM 2017
# Machine learning-based RBS predictor for the limonene pathway
# Main routines:
# doFullTest(R=1000)
# SummaryTest(R=1000)
# doLibrary(rbslib=1); doLibrary(rbslib=2)
# doFigure(rbslib=1); doFigure(rbslib=2)

getTrainingSet <- function(dataFile='data/rbs1/trainset.rbs1.v2.csv', colName='FC', colFeatStart=3, colFeatEnd=6,
                           AVERAGEINPUT=1, ODCORRECT=FALSE) {
                                        # Read training set for RBS1 Library 
                                        # colName: name of the column with the y values
                                        # colFeatStart .... TotalColumns-ColFeatEnd: Range of columns containing the features
                                        # AVERAGEINPUT:
                                        # 0: Put all values together, even if there are multiple identical RBS
                                        # 1: Average values per RBS
                                        # 2: Take a single RBS per average value
                                        # ODCORRECT: Fit a model that considers inhibition
                                        # Returns:
                                        # feat: Feature vector
                                        # y: target values
                                        # mm: initial data
   
    mm <- read.csv(dataFile, header=T, stringsAsFactors=F)
    labels <- c('rbs1', 'rbs2')
    y <- mm[,colName]
    if (ODCORRECT) {
        y <- y*log(mm$ODhar/mm$ODind)
    }
    feat <- mm[,colFeatStart:(dim(mm)[2]-colFeatEnd)]
    if (AVERAGEINPUT == 1) {
        ym <-  tapply(y, paste(mm$rbs1, mm$rbs2), mean)
        y <- ym[match(paste(mm$rbs1, mm$rbs2),names(ym))]
        mm$Label <- mm$Construct
        mm$Construct <- paste(mm$rbs1, mm$rbs2, sep='_')
    } else if (AVERAGEINPUT == 2) {
        y <-  tapply(y, paste(mm$rbs1, mm$rbs2), mean)
        feat <- feat[match(names(y), paste(mm$rbs1, mm$rbs2)),]
        mm <- mm[match(names(y), paste(mm$rbs1, mm$rbs2)),]
        mm$Labels <- tapply(mm$Construct, paste(mm$rbs1, mm$rbs2, sep='_'), function(x) {paste(sort(x), sep=',', collapse=",")})
        mm$Construct <- names(tapply(mm$Construct, paste(mm$rbs1, mm$rbs2, sep='_'), paste, sep=','))
    }
    return(list(y=y, feat=feat, mm=mm))
    
}

getTrainingSetLib2 <- function(dataFile='data/rbs2/trainset.rbs2.update.csv', colName='FC', colFeatStart=3, colFeatEnd=42,
                               AVERAGEINPUT=1, ODCORRECT=TRUE,
                               labels=c("mvaE.seq_rbs","mvaS.seq_rbs","mvaK1.seq_rbs","idi.seq_rbs")) {
                                        # Read training set for RBS2 library
                                        # colName: name of the column with the y values
                                        # colFeatStart .... TotalColumns-ColFeatEnd: Range of columns containing the features
                                        # AVERAGEINPUT:
                                        # 0: Put all values together, even if there are multiple identical RBS
                                        # 1: Average values per RBS
                                        # 2: Take a single RBS per average value
                                        # ODCORRECT: Fit a model that considers inhibition
                                        # Returns:
                                        # feat: Feature vector
                                        # y: target values
                                        # mm: initial data
   

    mm <- read.csv(dataFile, header=T, stringsAsFactors=F)
    y <- mm[,colName]
    if (ODCORRECT) {
        y <- y*log(mm$ODhar/mm$ODind)
    }
    feat <- mm[,colFeatStart:(dim(mm)[2]-colFeatEnd)]
    labs <- apply(mm[,labels], 1, paste, collapse="_")
    if (AVERAGEINPUT == 1) {
        ym <-  tapply(y, labs, mean)
        y <- ym[match(labs,names(ym))]
        mm$Labels <- mm$Construct
        mm$Construct <- labs
    } else if (AVERAGEINPUT == 2) {
        y <-  tapply(y, labs, mean)
        feat <- feat[match(names(y), labs),]
        mm <- mm[match(names(y), labs),]
        mm$Labels <- tapply(mm$Construct, labs, function(x) {paste(sort(x), sep=',', collapse=",")})
        mm$Construct <- names(tapply(mm$Construct, labs, paste, sep=','))
    }
    return(list(y=y, feat=feat, mm=mm))
    
}

compactTrainingSet <- function(dataset, ncomp) {
                                        # Select only ncomp principal components of the feature vector
    attach(dataset, warn.conflicts=FALSE)
    feat0 <- feat
    if (ncomp <= dim(feat)[1]) {
        feat <- predict(prcomp(feat), feat)[,1:ncomp]
    }
    dataset$feat <- feat
    return(dataset)
}

fit <- function(ks.kernel, ks.type, ks.scaled, dataset) {
                                        # Fit a model
    feat <- dataset[,1:(dim(dataset)[2]-1)]
    y <- dataset$y
    ks.boot <- ksvm(x=as.matrix(feat),
                    y=y, type=ks.type, kernel=ks.kernel, scaled=ks.scaled)
    y.pred <- predict(ks.boot)
    yy <- cbind(y, y.pred)
    return(yy)
}


rsq <- function(ks.kernel, ks.type, ks.scaled, mydata, perm, indices) {
                                        # Compute R2 of fitting
                                        # indices: cross-validation indices
                                        # perm: if True, shuffle only the output values (Permutation test)
    if (perm) {
        mydata$y <- mydata$y[indices]
    } else {
        mydata <- mydata[indices,]
    }
    yy <- fit(ks.kernel, ks.type, ks.scaled, mydata)
    return(cor(yy[,1], yy[,2])**2)
}

loo <- function(ks.kernel, ks.type, ks.scaled, dataset) {
                                        # Compute a LOO cross-validation
    feat <- dataset[,1:(dim(dataset)[2]-1)]
    y <- dataset$y
    s1 <- seq(1, length(y))
    yy <- c()
    for (i in seq(1, length(y))) {
        testSet <- i
        trainSet <- is.na(match(s1, testSet))
        ks.boot <- ksvm(x=as.matrix(feat[trainSet,]),
                        y=y[trainSet], type=ks.type, kernel=ks.kernel, scaled=ks.scaled)
        y.test <- y[i]
        y.test.pred <- predict(ks.boot, newdata=feat[testSet,])
        yy <- rbind(yy, c(y.test, y.test.pred))
    }
    return(yy)
}

cv <- function(ks.kernel, ks.type, ks.scaled, dataset, perc) {
                                        # Perfom a cross-validation

    feat <- dataset[,1:(dim(dataset)[2]-1)]
    y <- dataset$y
    s1 <- seq(1, length(y))
    mid.value <- floor(length(s1)*perc)
    testSet <- s1[1:mid.value]
    trainSet <- s1[(mid.value+1):length(s1)]
    ks.boot <- ksvm(x=as.matrix(feat[trainSet,]),
                    y=y[trainSet], type=ks.type, kernel=ks.kernel, scaled=ks.scaled)
    y.fit <- predict(ks.boot)
    lm1 <- lm(y ~ y.fit)

    y.test <- y[testSet]
    y.test.pred <- predict(ks.boot, newdata=feat[testSet,])
    y.pred <- predict(lm1, newdata=y.test.pred)
    yy <- cbind(y, y.pred)
    return(yy)
}


rmsecv <- function(ks.kernel, ks.type, ks.scaled, mydata, perm, perc, indices) {
                                        # Compute RMSECV
    if (perm) {
        mydata$y <- mydata$y[indices]
    } else {
        mydata <- mydata[indices,]
    }
    yy <- cv(ks.kernel, ks.type, ks.scaled, mydata, perc)
    return( sqrt(mean((yy[,1]-yy[,2])**2)) )
 
}

cv <- function(ks.kernel, ks.type, ks.scaled, dataset, perc) {
                                        # Cross-validation
                                        # perc: percentage of dataset used as test set
    feat <- dataset[,1:(dim(dataset)[2]-1)]
    y <- dataset$y
    s1 <- seq(1, length(y))
    mid.value <- floor(length(s1)*perc)
    testSet <- s1[1:mid.value]
    trainSet <- s1[(mid.value+1):length(s1)]
    ks.boot <- ksvm(x=as.matrix(feat[trainSet,]),
                    y=y[trainSet], type=ks.type, kernel=ks.kernel, scaled=ks.scaled)
    y.test <- y[testSet]
    y.test.pred <- predict(ks.boot, newdata=feat[testSet,])
    yy <- cbind(y.test, y.test.pred)
    return(yy)
}

rmsep <- function(ks.kernel, ks.type, ks.scaled, mydata, perm, indices) {
                                        # Compute RMSEP
    if (perm) {
        mydata$y <- mydata$y[indices]
    } else {
        mydata <- mydata[indices,]
    }
    feat <- mydata[,1:(dim(mydata)[2]-1)]
    y <- mydata$y
    ks.fit <-  ksvm(x=as.matrix(feat), y=y, type=ks.type, kernel=ks.kernel, scaled=ks.scaled)
    y <- y
    y.fit <- predict(ks.fit)
    lm1 <- lm(y ~ y.fit)
    y.pred <- predict(lm1)
    yy <- cbind(y, y.pred)
    return( sqrt(mean((yy[,1]-yy[,2])**2)) )
 
}


qsq <- function(ks.kernel, ks.type, ks.scaled, mydata, perm, indices) {
                                        # Compute Q2 of a LOO cross-validation
                                        # indices: cross-validation indices
                                        # perm: if True, shuffle only the output values (Permutation test)
    if (perm) {
        mydata$y <- mydata$y[indices]
    } else {
        mydata <- mydata[indices,]
    }
    yy <- loo(ks.kernel, ks.type, ks.scaled, mydata)
    return(cor(yy[,1],yy[,2])**2)
}


bst <- function(R1, fun, ks.kernel, ks.type, ks.scaled, mydata, perm, indices) {
                                        # Bootstrapping (with replacement) for some statistic (fun)
    if (perm) {
        mydata$y <- mydata$y[indices]
    }
    results.qsq <- boot(data=mydata, statistic=fun, R=R1, ks.kernel=ks.kernel, ks.type=ks.type, ks.scaled=ks.scaled)
    return(results.qsq$t0)
}


doBootstrapWithPermutation <- function(R=10, rbslib=1) {
                                        # Bootstrapping of the statistic with permutation test using boot package
    require('kernlab')
    require('boot')
    if (rbslib==1) {
           dataset <- getTrainingSet()
           dataset <- compactTrainingSet(dataset, ncomp=6)
           ks.kernel <- 'anovadot'
           ks.type <- 'eps-svr'
           ks.scaled <- T
       } else {
           dataset <- getTrainingSetLib2()
           dataset <- compactTrainingSet(dataset, ncomp=14)
           ks.kernel <- 'polydot'
           ks.type <- 'eps-svr'
           ks.scaled <- T
     }
    attach(dataset, warn.conflicts=FALSE)
    mydata <- data.frame(feat=feat, y=y)
    results.rmsecv <- boot(data=mydata, statistic=rmsecv, R=R, sim='permutation', ks.kernel=ks.kernel, ks.type=ks.type, ks.scaled=ks.scaled, perm=FALSE, perc=1/3)
    perm.rmsecv <- boot(data=mydata, statistic=rmsecv, R=R, sim='permutation', ks.kernel=ks.kernel, ks.type=ks.type, ks.scaled=ks.scaled, perm=TRUE, perc=1/3)
    results.rmsep <- boot(data=mydata, statistic=rmsep, R=R, sim='permutation', ks.kernel=ks.kernel, ks.type=ks.type, ks.scaled=ks.scaled, perm=FALSE)
    perm.rmsep <- boot(data=mydata, statistic=rmsep, R=R, sim='permutation', ks.kernel=ks.kernel, ks.type=ks.type, ks.scaled=ks.scaled, perm=TRUE)
    # Initial bootstrap
    results.qsq <- boot(data=mydata, statistic=qsq, R=R, sim='permutation', ks.kernel=ks.kernel, ks.type=ks.type, ks.scaled=ks.scaled, perm=FALSE)
    perm.qsq <- boot(data=mydata, statistic=qsq, R=R, sim='permutation',ks.kernel=ks.kernel, ks.type=ks.type, ks.scaled=ks.scaled, perm=TRUE)
    results.rsq <- boot(data=mydata, statistic=rsq, R=R, sim='permutation', ks.kernel=ks.kernel, ks.type=ks.type, ks.scaled=ks.scaled, perm=FALSE)
    perm.rsq <- boot(data=mydata, statistic=rsq, R=R, sim='permutation',ks.kernel=ks.kernel, ks.type=ks.type, ks.scaled=ks.scaled, perm=TRUE)
    return(list(qsq=results.qsq, perm.qsq=perm.qsq, rsq=results.rsq, perm.rsq=perm.rsq,
                rmsecv=results.rmsecv, perm.rmsecv=perm.rmsecv, rmsep=results.rmsep, perm.rmsep=perm.rmsep))
}

doFullTest <- function(R=10) {
                                        # Do a full cross-validation test for the two libraries
                                        # R: number of runs
                                        # Store results in perfomance+R+.RData
    reslib1 <- doBootstrapWithPermutation(R, rbslib=1)
    reslib2 <- doBootstrapWithPermutation(R, rbslib=2)
    save(reslib1, reslib2, file=paste("performance", R, ".RData", sep=''))
}

SummaryTest <- function(R=1000) {
                                        # Output summary information of the precomputed full test
                                        # R: number of runs
    load(paste("performance", R, ".RData", sep=''))
    R <- length(reslib1$qsq$t)
    cat(c('Lib1', 'R', R, 'Q2', reslib1$qsq$t0, 'p-value <=', max(1,length(which(reslib1$qsq$t0 < reslib1$perm.qsq$t)))/R, "\n"))
    cat(c('Lib1', 'R', R, 'R2', reslib1$rsq$t0, 'p-value <=', max(1,length(which(reslib1$rsq$t0 < reslib1$perm.rsq$t)))/R, "\n"))
    cat(c('Lib1', 'R', R, 'RMSECV', reslib1$rmsecv$t0, 'p-value <=', max(1,length(which(mean(reslib1$rmsecv$t) > reslib1$perm.rmsecv$t)))/R, "\n"))
    cat(c('Lib1', 'R', R, 'RMSEP', reslib1$rmsep$t0, 'p-value <=', max(1,length(which(reslib1$rmsep$t0 > reslib1$perm.rmsep$t)))/R, "\n"))
    cat(c('Lib2', 'R', R, 'Q2', reslib2$qsq$t0, 'p-value <=', max(1,length(which(reslib2$qsq$t0 < reslib2$perm.qsq$t)))/R, "\n"))
    cat(c('Lib2', 'R', R, 'R2', reslib2$rsq$t0, 'p-value <=', max(1,length(which(reslib2$rsq$t0 < reslib2$perm.rsq$t)))/R, "\n"))
    cat(c('Lib2', 'R', R, 'RMSECV', reslib2$rmsecv$t0, 'p-value <=', max(1,length(which(mean(reslib2$rmsecv$t) > reslib2$perm.rmsecv$t)))/R, "\n"))
    cat(c('Lib2', 'R', R, 'RMSEP', reslib2$rmsep$t0, 'p-value <=', max(1,length(which(reslib2$rmsep$t0 > reslib2$perm.rmsep$t)))/R, "\n"))
}


doPermutation <- function(R=10, rbslib=1, statistic=qsq) {
                                        # Permutation test of the given statistic using boot package 
       require('kernlab')
       require('boot')
       if (rbslib==1) {
           dataset <- getTrainingSet()
           dataset <- compactTrainingSet(dataset, ncomp=6)
           ks.kernel <- 'anovadot'
           ks.type <- 'eps-svr'
           ks.scaled <- T
       } else {
           dataset <- getTrainingSetLib2()
           dataset <- compactTrainingSet(dataset, ncomp=14)
           ks.kernel <- 'polydot'
           ks.type <- 'eps-svr'
           ks.scaled <- T
       }
       attach(dataset, warn.conflicts=FALSE)
       mydata <- data.frame(feat=feat, y=y)
       perm.qsq <- boot(data=mydata, statistic=statistic, R=R, sim='permutation', ks.kernel=ks.kernel, ks.type=ks.type, ks.scaled=ks.scaled, perm=TRUE)
       return(perm.qsq)
}


doPermutationTest <- function(R=1000, statistic=qsq) {
                                        # Perform a permutation test for both libraries
    t1 <- doPermutation(R=R, rbslib=1, statistic=statistic)
    t2 <- doPermutation(R=R, rbslib=2, statistic=statistic)
    return(list(t1=t1, t2=t2))
}


doBoostrappingPermutationTest <- function(R=1000, statistic=qsq) {
                                        # Perform a permutation test with bootstrapping for both libraries
    t1 <- doBootstrapWithPermutation(R=R, rbslib=1, statistic=statistic)
    t2 <- doBootstrapWithPermutation(R=R, rbslib=2, statistic=statistic)
    return(list(t1=t1, t2=t2))
}


doFigure <- function(rbslib=1) {
    require('kernlab')      # Generate test data and plot the figures
    if (rbslib==1) {
        ncomp <- 6
        ks.kernel <- 'anovadot'
        ks.type <- 'eps-svr'
        ks.scaled <- T
        dataset <- getTrainingSet()
        dataset <- compactTrainingSet(dataset, ncomp=ncomp)
        attach(dataset, warn.conflicts=FALSE)
        main.lab <- 'RBS library 1'
        datafile <- 'rbs1.RData'
    } else {
        ncomp <- 14
        ks.kernel <- 'polydot'
        ks.type <- 'eps-svr'
        ks.scaled <- T
        dataset <- getTrainingSetLib2()
        dataset <- compactTrainingSet(dataset, ncomp=ncomp)
        attach(dataset, warn.conflicts=FALSE)
        main.lab <- 'RBS library 2'
        datafile <- 'rbs2.RData'
    }
        # Common calculations
    lab.cex <- 0.75
    attach(dataset, warn.conflicts=FALSE)
    mydata <- data.frame(feat=feat, y=y)
    yy0 <- fit(ks.kernel, ks.type, ks.scaled, mydata)
    yy <- loo(ks.kernel, ks.type, ks.scaled, mydata)
    row.names(yy) <- mm$Construct
    row.names(yy0) <- mm$Construct
    plot(yy[,1], yy[,2], xlab='Production', ylab='Predicted', pch='',
         main=paste(main.lab,
             'R2=', format(cor(yy0[,1], yy0[,2])**2, digits=2),
             'Q2=', format(cor(yy[,1], yy[,2])**2, digits=2)), xlim=c(0,2), ylim=c(0,2))
    text(yy[,1], yy[,2], row.names(yy), cex=lab.cex)
    grid()
    yy1 <- data.frame(obsv=yy[,1], pred=yy[,2], label=mm$Label, rbs=rownames(yy))
    yy <- yy1
    yy1 <- data.frame(obsv=yy0[,1], pred=yy0[,2], label=mm$Label, rbs=rownames(yy0))
    yy0 <- yy1
    save(yy0, yy, file=datafile)
}


PlotFigs <- function(rbslib=1, GG=FALSE) {
                                        # Plot the figures from stored data
    if (rbslib==1) {
        load('rbs1.RData')
        main.lab <- 'RBS library 1'
    }
    else {
        load('rbs2.RData')
        main.lab <- 'RBS library 2'
    }
    lab.cex <- 0.75
    if (GG==FALSE) {
        plot(yy$obsv, yy$pred, xlab='Production [FC]', ylab='Predicted [FC]', pch='',
             main=paste(main.lab,
                 'R2=', format(cor(yy0$obsv, yy0$pred)**2, digits=2),
                 'Q2=', format(cor(yy$obsv, yy$pred)**2, digits=2)), xlim=c(0,2), ylim=c(0,2))
        text(yy$obsv, yy$pred, yy$rbs, cex=lab.cex)
        abline(a=0, b=1, lt=2)
        grid()
    } else {
        # Using ggplot2
        require(ggplot2)
        p <- ggplot(data.frame(production=yy$obsv, predicted=yy$pred, nam=yy$rbs), aes(production, predicted)) +
            xlim(0,2.0) + ylim(0,2.0) + labs( x = "Production [FC]", y = "Predicted [FC]") +
                geom_text(aes(label=nam), size=3) + geom_smooth(method= "lm")
        print(p)
        ggsave(paste("rbs", rbslib, ".png", sep=''), width = 8, height = 8)
    }
}


doLibrary <- function(rbslib=1) {
                                        # Generate predictions for the full library
    require('kernlab')
    if (rbslib==1) {
        ncomp <- 6
        ks.kernel <- 'anovadot'
        ks.type <- 'eps-svr'
        ks.scaled <- T
        dataset <- getTrainingSet()
        attach(dataset, warn.conflicts=FALSE)
        feat0 <- feat
        dataset <- compactTrainingSet(dataset, ncomp=ncomp)
        attach(dataset, warn.conflicts=FALSE)
        nlib <- read.csv('data/rbs1/fullset.rbs1.v2.csv', header=F)
        featLib <- nlib[,colFeat:dim(nlib)[2]]
        labels <- c('rbs1', 'rbs2')
        seqCols <- c(3,4)
    } else {
        ncomp <- 14
        ks.kernel <- 'polydot'
        ks.type <- 'eps-svr'
        ks.scaled <- T
        colFeat <- 9
        dataset <- getTrainingSetLib2()
        attach(dataset, warn.conflicts=FALSE)
        feat0 <- feat
        dataset <- compactTrainingSet(dataset, ncomp=14)
        attach(dataset, warn.conflicts=FALSE)
        nlib <- read.csv('data/rbs2/newfullset.v2.csv', header=F)
        featLib <- nlib[,colFeat:dim(nlib)[2]]
        labels <- c("mvaE.seq", "mvaS.seq", "mvaK1.seq", "idi.seq")
        seqCols <- seq(1, 4)
    }
    # Common calculations
    ks <- ksvm(x=as.matrix(feat),
               y=y, type=ks.type, kernel=ks.kernel, scaled=ks.scaled)
    names(featLib) <- names(feat0)
    featLib <- predict(prcomp(feat0), featLib)[,1:ncomp]
    libSet <- seq(1, dim(nlib)[1])
    step <- 1e3
    ylib <- c()
    for (k in seq(1, length(libSet), by=step)) {
        if ((k+step-1)>length(libSet)) {
            libSet1 <- libSet[k:length(libSet)]
        } else {
            libSet1 <- libSet[k:(k+step-1)]
        }
        ylib <- c(ylib, predict(ks, newdata=featLib[libSet1,]))
    }

    res <- cbind(nlib[, 1:(colFeat-1)], ylib)
    seqLabels <- apply(res[,seqCols], 1, paste, collapse='_')
    rbsLabels <- apply(mm[,labels], 1, paste, collapse='_')
    ypred <- tapply(res$ylib, seqLabels, mean)
    yexp <- tapply(y, rbsLabels, mean)
    rbsid <- tapply(mm$Construct, rbsLabels, function(x) {as.character(x[1])})
    yfull <- ypred
    yfull[names(yfull)] <- NA
    yfull[names(yexp)] <- yexp
    yfull2 <- ypred
    yfull2[names(yfull2)] <- NA
    yfull2[names(rbsid)] <- rbsid

    dd <- data.frame(ypred=ypred, yexp=yfull, rbsid=yfull2)
    dd <- dd[order(dd$ypred),]
 
    res <- cbind(res, ylib)

    if (rbslib==1) {
        names(res) <- c("GPPS.seq","limS.seq","GPPS.rbs","limS.rbs","lim_svm", "lim_eff")
        write.csv(dd, 'data/rbs1/rbslib1_pred.paper0.csv')
        write.csv(res[order(res[,dim(res)[2]]),], 'data/rbs1/rbslib1_pred.paper.csv',quote=F,row.names=F)
    } else {
        names(res) <- c("mvaE.seq","mvaS.seq","mvaK1.seq","idi.seq","mvaE.rbs","mvaS.rbs","mvaK1.rbs","idi.rbs","lim_svm", "lim_eff")
        write.csv(dd, 'data/rbs2/rbslib2_pred.paper0.csv')
        write.csv(res[order(res[,dim(res)[2]]),], 'data/rbs2/rbslib2_pred.paper.csv', quote=F,row.names=F)
    }

}

ranktest <- function() {
    # Perform a Wilcoxon rank test for the two MVA libraries 
    dat <- getTrainingSetLib2()
    m1 <- dat$mm
    m2 <- read.csv('data/rbs2.2/results_RBS_MVA2_ods.csv')
    ix1 <- grep('MVARBS', m2$RBS)
    ix2 <- grep('MVARBS41_', m2$RBS, invert=T)
    ix <- intersect(ix1, ix2)
    m2 <- m2[ix,]
    ix3 <- m2$FC > 0.7
    m1$FC <- m1$FC/m1$ODharv
    m2$FC <- m2$FC/m2$OD_harvest
    plot(density(m2$FC), xlim=c(0,3), ylim=c(0,2), main='', xlab='FC')
    lines(density(m2$FC[ix3]), col='blue')
    lines(density(m1$FC), col='red')
    cat('Test with outliers')
    print(wilcox.test(m2$FC, m1$FC, alt='greater'))
    cat('Test without outliers < 0.7')
    print(wilcox.test(m2$FC[ix3], m1$FC, alt='greater'))
    legend('topleft', legend=c('All values', 'Filtered outliers < 0.7', 'First library'), col=c('black', 'blue', 'red'), lw=1)
    grid()
}

tablogs <- function() {
    dat1 <-  getTrainingSet()
    su1 <- summary(dat1$mm$FC)
    su1.q1 <- su1[2]
    su1.q3 <- su1[5]
    ix1 <- dat1$mm$FC <= su1.q1
    ix2 <- dat1$mm$FC >= su1.q3
    write.table(dat1$mm$rbs1[ix1], file='data/rbs1/seqlogos/lo-gpps.txt', quote=F, row.names=F, col.names=F)
    write.table(dat1$mm$rbs2[ix1], file='data/rbs1/seqlogos/lo-lims.txt', quote=F, row.names=F, col.names=F)
    write.table(dat1$mm$rbs1[ix2], file='data/rbs1/seqlogos/hi-gpps.txt', quote=F, row.names=F, col.names=F)
    write.table(dat1$mm$rbs2[ix2], file='data/rbs1/seqlogos/hi-lims.txt', quote=F, row.names=F, col.names=F)
    dat2 <- getTrainingSetLib2()
    su2 <- summary(dat2$mm$FC)
    su2.q1 <- su2[2]
    su2.q3 <- su2[5]
    ix1 <- dat2$mm$FC <= su2.q1
    ix2 <- dat2$mm$FC >= su2.q3
    write.table(dat2$mm$mvaE.seq_rbs[ix1], file='data/rbs2/seqlogos/lo-mvaE.txt', quote=F, row.names=F, col.names=F)
    write.table(dat2$mm$mvaS.seq_rbs[ix1], file='data/rbs2/seqlogos/lo-mvaS.txt', quote=F, row.names=F, col.names=F)
    write.table(dat2$mm$mvaK1.seq_rbs[ix1], file='data/rbs2/seqlogos/lo-mvaK1.txt', quote=F, row.names=F, col.names=F)
    write.table(dat2$mm$idi.seq_rbs[ix1], file='data/rbs2/seqlogos/lo-idi.txt', quote=F, row.names=F, col.names=F)
    write.table(dat2$mm$mvaE.seq_rbs[ix2], file='data/rbs2/seqlogos/hi-mvaE.txt', quote=F, row.names=F, col.names=F)
    write.table(dat2$mm$mvaS.seq_rbs[ix2], file='data/rbs2/seqlogos/hi-mvaS.txt', quote=F, row.names=F, col.names=F)
    write.table(dat2$mm$mvaK1.seq_rbs[ix2], file='data/rbs2/seqlogos/hi-mvaK1.txt', quote=F, row.names=F, col.names=F)
    write.table(dat2$mm$idi.seq_rbs[ix2], file='data/rbs2/seqlogos/hi-idi.txt', quote=F, row.names=F, col.names=F)
}
