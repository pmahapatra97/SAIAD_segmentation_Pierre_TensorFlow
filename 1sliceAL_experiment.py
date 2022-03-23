# -*- coding: utf-8 -*-

from train_test_unet import *

import constant
import data


# set randomness
np.random.seed(1)
tf.random.set_seed(2)



if __name__ == '__main__':
    
    #-----------------------------------------------#
    #           Initialisation                      #
    #-----------------------------------------------#
    
    #-------------get args from parser--------------#
    args = get_from_parser(verbose=True)

    #----------------init model path----------------#
    path_save_model_empty = '../save_model/unet_model_{}_{{}}_{}.h5'.format(args.experiment, args.structure)
    path_old_save_model_empty = '../save_model/unet_model_{}_{{}}_{}.h5'.format(args.ovassion_exp, args.structure)

    #--------------init results array---------------#
    results_patient = []
    results_slices = []

    #---------store experiment informations---------#
    store_experiment(args)
    
    for patient in args.patients :
        path_res = constant.RESULTS_DIR.format(args.experiment,patient)
        # set randomness
        np.random.seed(1)
        tf.random.set_seed(2)
        
        print("Patient : "+patient)
        args.patient = patient
        #-----------------------------------------------#
        #                 Get data                      #
        #-----------------------------------------------#
        print("GET DATA")
        
        data_train_test = data.get_data(patient, args)
        try:
            print(data_train_test.train_idx)
        except:
            print("NO train_idx")

        shiftList = list(range(data_train_test.nbSlices))
        for shift in shiftList:
            tf.random.set_seed(2)
        
            data_train_test.set_shift(shift)
            # -----------------------------------------------#
            #                 Create model                  #
            # -----------------------------------------------#
            print("MODEL INITIALISATION")

            #------------------from scratch-----------------#
            print("Create Model")
            path_old_save_model = path_old_save_model_empty.format(patient)
            model = init_unet(path_old_save_model, args)
            
            #-----------------------------------------------#
            #                 Train model                   #
            #-----------------------------------------------#
            
            print("TRAIN MODEL")
            #---------------------train---------------------#
            model, history = train(model, data_train_test, args)
            training_plot_path = '../plots/training_plot_{}_{}_{}_{}epochs_{}lr_{}bs_{}_shift'.format(args.experiment,patient,args.structure,args.epochs,args.learning_rate,args.batch_size,shift)
            save_training_plot(history, training_plot_path)
            
            #---------------------save----------------------#
            # path_save_model = path_save_model_empty.format(patient)
            # model.save(path_save_model)

            # -----------------------------------------------#
            #                 Predictions                   #
            # -----------------------------------------------#
            print("PREDICTIONS")
            

            # #-------------prediction------------------------#
            print("Prediction")
            test_predict(model, data_train_test, path_res)

            #------prediction with monte carlo dropout------#
            # print("Monte-Carlo dropout prediction")
            # test_predict_dropout(model, data_train_test, path_res)

            # -----------------------------------------------#
            #             Compute metrics                   #
            # -----------------------------------------------#
            print("COMPUTE METRICS")

            # -----------------get results-------------------#
            print("Get Predictions")
            get_predict(data_train_test, path_res)

            # --------------compute metrics------------------#
            print("Compute metrics")
            logPath =  '/logs/resultDiceIU_{}.txt'.format(shift)
            metric_results_patient, metric_results_slices = test_metrics(data_train_test, path_res,logPath)
            for me in metric_results_patient:
                me["shift"]=shift
            results_patient += metric_results_patient
            # results_slices += metric_results_slices
            tf.keras.backend.clear_session()
        
        
    #-----------------------------------------------#
    #              Store results                    #
    #-----------------------------------------------#
    print("Store results csv")
    df_res = pd.DataFrame(results_patient).set_index(["Patient","structure"]).apply(pd.to_numeric)
    df_res.reset_index(inplace = True)
    structures = df_res.structure.unique()
    df_res_mean = df_res.copy()
    for structure in structures:
        df_res.loc[df_res.index.max()+1]= ["Mean",structure] + list(df_res_mean[df_res_mean.structure==structure].mean())
    df_res.set_index(["Patient","structure"], inplace = True)
    # df_res.loc["Mean"]= df_res.mean()
    df_res.to_csv(constant.CSV_FILE.format(args.experiment))
    df_res_slices = pd.DataFrame(results_slices).set_index(["Patient","idx","structure"]).apply(pd.to_numeric)
    df_res_slices.to_csv(constant.CSV_FILE_SLICES.format(args.experiment))
