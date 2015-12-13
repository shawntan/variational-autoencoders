from __future__ import print_function
import time
import numpy as np
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def run(data_iterator,train_fun,validation_score,
        save_best_params,load_best_params,
        max_epochs=1000,
        patience=5000,patience_increase=2,
        improvement_threshold=0.995,
        validation_frequency=None):

    if validation_frequency is None:
        print(bcolors.OKBLUE + 'Counting minibatches in epoch to set validation_frequency' + bcolors.ENDC)
        minibatch_count = sum(1 for _ in data_iterator())
        print(bcolors.OKBLUE + 'Minibatch count: %d'%minibatch_count + bcolors.ENDC)
        validation_frequency = min(minibatch_count,patience/2)

    print(bcolors.OKBLUE)
    print('max_epochs:',max_epochs)
    print('patience:',patience)
    print('patience_increase:',patience_increase)
    print('improvement_threshold:',improvement_threshold)
    print('validation_frequency:',validation_frequency)
    print(bcolors.ENDC)

    best_validation_loss = validation_score()
    save_best_params()

    done_looping = False
    epoch = 0
    iterations = 0
    print(bcolors.BOLD)
    print("Starting training....")
    print(bcolors.ENDC)
    print(bcolors.OKGREEN)
    print('Best score:',best_validation_loss)
    print(bcolors.ENDC)
    for epoch in xrange(max_epochs):
        if patience <= iterations: break
        start_time = time.clock()
        print(bcolors.BOLD + 'Starting epoch %d'%(epoch+1) + bcolors.ENDC)
        for batch in data_iterator():
            scores = train_fun(batch)
            #print(np.array(scores))
            if np.isnan(scores).any():
                print("NaN")
                exit()
            # iteration number. We want it to start at 0.
            # note that if we do `iter % validation_frequency` it will be
            # true for iter = 0 which we do not want. We want it true for
            # iter = validation_frequency - 1.
            if (iterations + 1) % validation_frequency == 0:
                print(bcolors.OKBLUE + 'Running validation...' + bcolors.ENDC)
                this_validation_loss = validation_score()
                print(bcolors.OKBLUE + 'Validation loss: %0.5f'%this_validation_loss + bcolors.ENDC)

                if this_validation_loss < best_validation_loss:
                    print(bcolors.OKGREEN)
                    print('improvement seen')
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iterations * patience_increase)
                    print('saving model')
                    best_validation_loss = this_validation_loss
                    save_best_params()
                    print(bcolors.OKGREEN)
                    print('Best score:',best_validation_loss)
                    print(bcolors.ENDC)

                print(bcolors.OKBLUE + '%d iterations left'%(patience - iterations) + bcolors.ENDC)
            iterations += 1

    print(bcolors.BOLD + bcolors.OKGREEN + "Final result: %0.5f"%best_validation_loss + bcolors.ENDC)

    load_best_params()




