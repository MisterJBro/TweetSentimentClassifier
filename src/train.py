# Script for training models in the ensemble
from models import ModelBert, ModelRoberta, ModelXLNet, ModelDistilBert, ModelAlbert, ModelBertweet, ModelTwitterRoberta
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':
    # Variables
    BATCH_SIZE = 1
    LEARNING_RATE = 2e-5

    # Initialize model (Bert, Roberta, XLNet, DistilBert, Albert, Bertweet, TwitterRoberta)
    model_names = ['Bert', 'Roberta', 'XLNet', 'DistilBert', 'Albert', 'Bertweet', 'TwitterRoberta']

    for name in model_names:
        model = eval(f'Model{name}(lr=LEARNING_RATE, batch_size=BATCH_SIZE)')

        # Validation loss checkpoint
        checkpoint_callback = ModelCheckpoint(
            monitor='val_f1',
            dirpath=f'../checkpoints',
            filename=f'{name.upper()}_{{epoch:02d}}_{{val_f1:.2f}}',
            save_top_k=1,
            mode='max',
        )

        # Initialize trainer and start training
        trainer = Trainer(
            gpus=1,
            max_epochs=4,
            callbacks=[checkpoint_callback],
            accumulate_grad_batches=16,
        )
        trainer.fit(model)