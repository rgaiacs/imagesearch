import sys, getopt
import run
 
def main(argv):
    
    path_database = ''
    path_cnn_trained = ''
    path_folder_retrieval = ''

    feature_extraction_method = ''
    distance = ''
    searching_method = ''
    number_of_images = 0
    list_of_parameters = []
    
    try:
      opts, args = getopt.getopt(argv,"hd:c:r:f:s:p:n:m:")
    except getopt.GetoptError:
      print ('cbir_cl.py -d <path_database> -c <path_cnn_trained> -r <path_folder_retrieval> -f <feature_extraction_method> -s <distance-similarity metric> -n <number_of_images> -m <list_of_parameters>')
      sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print ('cbir_cl.py -d <path_database> -c <path_cnn_trained> -r <path_folder_retrieval> -f <feature_extraction_method> -s <distance-similarity metric> -n <number_of_images> -m <list_of_parameters>')
            sys.exit()
        elif opt == '-d':
            path_database = arg 
        elif opt == '-c':
            path_cnn_trained = arg
        elif opt == '-r':
            path_folder_retrieval = arg
        elif opt == '-f':
            feature_extraction_method = arg
        elif opt == '-s':
            distance = arg
        elif opt == '-p':
            searching_method = arg
        elif opt == '-n':
            number_of_images = int(float(arg))
        elif opt == '-m':
            parameters = arg.split(',')
            for i in parameters:
                list_of_parameters.append(i)
 
    run.run_command_line(path_database,path_folder_retrieval,path_cnn_trained,feature_extraction_method,distance,number_of_images,list_of_parameters)
    
if __name__ == "__main__":
    main(sys.argv[1:])
       