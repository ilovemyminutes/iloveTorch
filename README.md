# iloveTorch
documents of my own of pytorch, deep learning framework for studying 

- `.named_parameters(prefix: str='', recurse: bool=True) -> Iterator[Tuple[str, torch.Tensor()]]`: 모듈의 파라미터를 이름을 붙여 yield하는 제너레이터를 반환

## Overall Configuration

- `torch`: 메인 네임스페이스. `Tensor` 등 다양한 수학 함수를 포함. NumPy와 유사한 방식으로 작동.
- `torch.Autograd`: 자동 미분 도구를 지원하는 모듈. 자동미분을 on/off하는 `enable_grad`, `no_grad` 등 지원
- `torch.nn`: 신경망 구축을 위한 다양한 데이터 구조/레이어가 정의된 모듈. Layer, Activation Function, Loss Function 등 포함
- `torch.optim`: 파라미터 최적화 알고리즘을 지원하는 모듈

- `torch.utils.data`: SGD의 반복 연산 실행시 활용되는 미니배치용 유틸리티 함수가 포함된 모듈
- `torch.onnx`: onnx(open neural network exchange)의 포맷으로 모델을 export할 떄 사용. onnx 포맷을 활용하면 서로 다른 딥러닝 프레임워크 간 모델 공유 가능
- `torchvision.datasets`: 파이토치에서 지원하는 데이터셋을 불러오기 위한 모듈
- `torch.utils.data.DataLoader`: 