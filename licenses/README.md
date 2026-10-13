# Code and model licenses

The FluxSR project code is licensed under [Apache-2.0](../LICENSE).
The FLUX.1-dev base model, FluxSR LoRA weights, and bundled fixed T5/CLIP
text-embedding tensors are subject to the
[FLUX.1 \[dev\] Non-Commercial License](FLUX.1-dev-LICENSE.md).
The Apache-2.0 code license does not relicense those model assets. This
includes the embeddings distributed in the `weights/fluxsr/` model
package.

`FLUX.1-dev-LICENSE.md` contains the license supplied with the base model
used to train the FluxSR adapter. The upstream document is available on
the [BFL license page](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).

For redistribution, include the supplied model license and the required
attribution and modification notices in [NOTICE](../NOTICE). In
particular, the model license's Distribution section requires a copy of
the license, the BFL attribution notice, a statement that the model was
modified, and no suggestion of BFL endorsement. Recipients obtain model
rights directly from BFL under that license.

The standalone `weights/fluxsr/` model package includes its model license
and attribution notice alongside the LoRA adapter, fixed embeddings,
configuration, prompts JSON, and model card. Upload these files together
when publishing the model package. Keep the project code license separate
from the model license.

Official sources:

- [FLUX.1-dev model and access conditions](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [Current FLUX.1-dev model license](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)
- [BFL reference-code Apache-2.0 license](https://github.com/black-forest-labs/flux/blob/main/LICENSE)

## Included source components

- The differentiable JPEG implementation in `dataset/utils.py` is adapted from
  [DiffJPEG](https://github.com/mlomnitz/DiffJPEG); see
  [its MIT license](DiffJPEG-LICENSE.txt).
- Noise routines in `dataset/degradations.py` reference
  [scikit-image](https://github.com/scikit-image/scikit-image); the applicable
  [BSD-3-Clause notice](scikit-image-LICENSE.txt) is retained.
- The degradation utilities build on
  [BasicSR](https://github.com/XPixelGroup/BasicSR); see
  [its Apache-2.0 license](BasicSR-LICENSE.txt).

These components retain their upstream licenses. Their code is adapted for
the FluxSR offline-pair training pipeline.
