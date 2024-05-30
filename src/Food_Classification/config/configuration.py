from Food_Classification.constants import *
from Food_Classification.utils.common import read_yaml,create_directory
from Food_Classification.entity.config_entity import DataIngestionConfig, PrepareBasemodelConfig, TrainingConfig, DataTransformConfig, EvaluationConfig, ModelPusherConfig
class ConfigurationManager:
    def __init__(self,
                 config_file_path = CONFIG_FILE_PATH,
                 params_file_path = PARAMS_FILE_PATH):
        
        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARAMS_FILE_PATH)

        create_directory([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
            config = self.config.data_ingestion

            create_directory([config.root_dir])

            data_ingestion_config = DataIngestionConfig(
                root_dir= config.root_dir,
                source_URL= config.source_url,
                local_data_file=config.local_data_file,
                unzip_dir=config.unzip_dir)
            
            return data_ingestion_config

    def get_base_model_config(self) -> PrepareBasemodelConfig:
        config = self.config.prepare_base_model
        params = self.params

        create_directory([config.root_dir])

        prepare_base_model_cofig = PrepareBasemodelConfig(
                                                        root_dir= Path(config.root_dir),
                                                        base_model_dir= Path(config.base_model_path),
                                                        updated_base_model= Path(config.updated_base_model),
                                                        params_image_size=  params.IMAGE_SIZE,
                                                        params_device= params.DEVICE,
                                                        params_weight= params.WEIGHTS,
                                                        params_classes= params.CLASSES,
                                                        p= params.p
                                                        )
        
        return prepare_base_model_cofig


    def get_DataTransformConfig(self) -> DataTransformConfig:
        config = self.config.data_transforms
        create_directory([config.root_dir])
        train_dir = os.path.join(self.config.data_ingestion.unzip_dir,'food_40_percent','train')
        test_dir = os.path.join(self.config.data_ingestion.unzip_dir,'food_40_percent','test')
        train_transformed_file = Path(config.TRAIN_TRANSFORMS_FILE)
        test_transformed_file = Path(config.TEST_TRANSFORMS_FILE)

        TransformationConfig = DataTransformConfig(root_dir= config.root_dir,
                                                   train_dir=Path(train_dir),
                                                   test_dir=Path(test_dir),
                                                   train_transforms_file=train_transformed_file,
                                                   test_transforms_file=test_transformed_file,
                                                   blur= {"kernel_size": KERNEL_SIZE,
                                                          "sigma": SIGMA},
                                                   affine={"degrees": DEGREES,
                                                           "translate":TRANSLATE,
                                                           "scale": SCALE,
                                                           "shear": SHEAR},
                                                   color_transform={"brightness":BRIGHTNESS,
                                                                    "contrast":CONTRAST,
                                                                    "saturation":SATURATION,
                                                                    "hue":HUE},
                                                    spatial_transform= {'vertical_flip': VERTICLE_FLIP,
                                                                        'resize': RESIZE,
                                                                        'center_crop': CENTER_CROP,
                                                                        'rotation': RANDOMROTATION
                                                                        },
                                                    normalize= {'mean': NORMALIZE_MEAN,
                                                                'std': NORMALIZE_STD},
                                                    data_loader_params={'batch_size': self.params.BATCH_SIZE,
                                                                        'shuffle': self.params.SHUFFLE,
                                                                        'num_workers': NUM_WORKERS,
                                                                        'pin_memory': PIN_MEMORY,
                                                                         })
        return TransformationConfig




    def get_training_config(self) -> TrainingConfig:
        
        config = self.config.model_training
        create_directory([config.root_dir])
        model_config = self.config.prepare_base_model
        TrainConfig = TrainingConfig(root_dir= config.root_dir,
                                     loss_function=LOSS_FUNCTION(),
                                     learning_rate= self.params.LEARNING_RATE,
                                     epochs= self.params.EPOCHS,
                                     device= self.params.DEVICE,
                                     classes= self.params.CLASSES,
                                     #schedular_params= {"step_size": self.params.STEP_SIZE,
                                                        #"gamma": self.params.GAMMA},
                                     schedular_params= {"mode": self.params.MODE,
                                                        "factor":self.params.FACTOR,
                                                        "patience": self.params.PATIENCE,
                                                        "verbose": self.params.VERBOSE},
                                     bentoml_model_name= "food_classification_model",
                                     train_transform_key= "TRAIN_TRANSFORM_KEY")
        
        return TrainConfig


    def get_evaluation_config(self) -> EvaluationConfig:
        Eval_Config = EvaluationConfig(loss_function = LOSS_FUNCTION(),
                                        device= self.params.DEVICE)
        
        return Eval_Config
    
    def get_model_pusher_config(self) -> ModelPusherConfig:
        PusherConfig = ModelPusherConfig(bentoml_model_name=BENTOML_MODEL_NAME,
                                          bentoml_service_name=BENTOML_SERVICE_NAME,
                                          train_transform_key=TRAIN_TRANSFORM_KEY,
                                          bentoml_ecr_image=BENTOML_ECR_URI)
         
        return PusherConfig
         

