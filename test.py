import mxnet as mx
from mxnet import image, gluon

ctx = mx.gpu() if mx.context.num_gpus() >0 else mx.cpu()

gluon.utils.download('https://s3.amazonaws.com/onnx-mxnet/examples/super_res_input.jpg')

img = image.imread('super_res_input.jpg').astype('float32')/255
img = mx.nd.transpose(img, (2,0,1))

gluon.utils.download('https://raw.githubusercontent.com/WolframRhodium/Super-Resolution-Zoo/master/ARAN/aran_c0_s1_x4-symbol.json')
gluon.utils.download('https://raw.githubusercontent.com/WolframRhodium/Super-Resolution-Zoo/master/ARAN/aran_c0_s1_x4-0000.params')

net = gluon.SymbolBlock.imports("aran_c0_s1_x4-symbol.json",['data'], "aran_c0_s1_x4-0000.params")
net.collect_params().reset_ctx(ctx)

output = net(img.expand_dims(0).as_in_context(ctx))
output = mx.nd.squeeze(output)
output = (mx.nd.transpose(output, (1,2,0))*255).astype('uint8')

from PIL import Image
img = Image.fromarray(output.asnumpy())
img.save('ARAN_4x.jpg')