import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np


class Evaluator(object):

    def __init__(self, device, model, dataset):
        self._device = device
        self._model = model
        self._dataset = dataset

    def reconstruct(self):
        self._model.eval()

        (self._valid_originals, _) = next(iter(self._dataset.validation_loader))
        self._valid_originals = self._valid_originals.to(self._device)

        vq_output_eval = self._model._pre_vq_conv(self._model._encoder(self._valid_originals))
        _, valid_quantize, _, _ = self._model._vq_vae(vq_output_eval)
        self._valid_reconstructions = self._model._decoder(valid_quantize)

        (train_originals, _) = next(iter(self._dataset.training_loader))
        train_originals = train_originals.to(self._device)
        _, self._train_reconstructions, _, _ = self._model._vq_vae(train_originals)

    def save_original_images_plot(self, path):
        self._save_image(make_grid(self._valid_originals.cpu()+0.5), path)

    def save_validation_reconstructions_plot(self, path):
        self._save_image(make_grid(self._valid_reconstructions.cpu().data)+0.5, path)

    def save_embedding_plot(self, path):
        try:
            import umap
        except ImportError:
            raise ValueError('umap-learn not installed')

        map = umap.UMAP(
            n_neighbors=3,
            min_dist=0.1,
            metric='euclidean'
        )

        projection = map.fit_transform(self._model.vq_vae.embedding.weight.data.cpu())

        fig = plt.figure()
        plt.scatter(projection[:,0], projection[:,1], alpha=0.3)
        fig.savefig(path)
        plt.close(fig)

    def _save_image(self, img, path):
        npimg = img.numpy()
        plt.imsave(path, np.transpose(npimg, (1, 2, 0)))
