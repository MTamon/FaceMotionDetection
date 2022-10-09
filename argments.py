from argparse import ArgumentParser

def add_args(parser: ArgumentParser):
    parser.add_argument('--file', default=None, type=str, help='vide file path which is used experiment')
    parser.add_argument('--file-codec', default='mp4v', type=str, help='video file codec')
    parser.add_argument('--output-path', default='./get_ability/prot_data/out')
    parser.add_argument('--output-name', default='out.mp4')
    parser.add_argument('--temp-path', default='./get_ability/prot_data/divis')
    parser.add_argument('--use-all-lm', default=False, action='store_true', help='set program mode: use all landmarks')
    parser.add_argument('--used-lm-number', default=1, type=int, help="select used landmark's number")
    parser.add_argument('--thickness', default=1, type=int, help="one of mediapipe's DrawingSpec class argment")
    parser.add_argument('--circle-radius', default=2, type=int, help="one of mediapipe's DrawingSpec class argment")
    parser.add_argument('--static-image-mode', default=False, type=bool, help="one of mediapipe's FaceMesh class argment")
    parser.add_argument('--max-num-faces', default=1, type=int, help="one of mediapipe's FaceMesh class argment")
    parser.add_argument('--refine-landmarks', default=False, type=bool, help="one of mediapipe's FaceMesh class argment")
    parser.add_argument('--min-detection-confidence', default=0.5, type=float, help="one of mediapipe's FaceMesh class argment")
    parser.add_argument('--min-tracking-confidence', default=0.5, type=float, help="one of mediapipe's FaceMesh class argment")
    parser.add_argument('--model-selection', default=0, type=int, help="one of mediapipe's FaceDetection class argment")
    parser.add_argument('--measure-time', default=False, action='store_true', help='print process time to console')
    parser.add_argument('--display-progress', default=False, action='store_true', help='print progress to console')
    
    return parser

def get_args():
    parser = ArgumentParser('This program is experimental code for getting ability to impliment Face Mesh.')
    parser = add_args(parser)
    
    return parser.parse_args()