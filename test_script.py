import argparse
import sys
import pytorch_lightning as pl
from pathlib import Path
from clip import clip
from pytorch_lightning import seed_everything

sys.path.append(str(Path(__file__).parent.parent / 'thesislib'))
from thesislib.datamodules import MSRVTTDataModule
from thesislib.models import RetrievalCLIP

datamodules = {
    'msrvtt': MSRVTTDataModule,
}

visual_embedding_dims = {
    'ViT-B/32': 768,
    'ViT-B/16': 768,
    'ViT-L/14': 1024,
}
text_embedding_dims = {
    'ViT-B/32': 512,
    'ViT-B/16': 512,
    'ViT-L/14': 512,
}


def main(args):
    seed_everything(seed=args.seed, workers=True)

    datamodule = datamodules[args.dataset](
        data_root=args.data_root,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        nr_frames=args.nr_context_frames,
        prompt_prefix=args.prompt_prefix,
        fps=args.fps,
    )

    if not args.disable_vca:
        vca_settings = {
            'vca_mode': args.vca_mode,
            'nr_output_vectors': args.vca_length,
            'vector_dim': visual_embedding_dims[args.architecture],
            'video_resolution': args.video_resolution,
            'input_type': args.video_input_type,
            'pretrained': args.pretrained,
            'temporal_permutation': args.temporal_permutation,
            'image_sample_mode': args.image_sample_mode,
        }
    else:
        vca_settings = None

    if not args.disable_tca:
        tca_settings = {
            'tca_mode': args.tca_mode,
            'nr_output_vectors': args.vca_length,
            'vector_dim': text_embedding_dims[args.architecture],
            'insertion_mode': args.tca_insertion,
        }
    else:
        tca_settings = None

    if args.from_checkpoint:
        retrieval_clip = RetrievalCLIP.load_from_checkpoint(
            Path(__file__).parent / 'checkpoints' / args.ckpt_file_name
        )
        print(f"loaded {args.ckpt_file_name} with backbone {args.architecture}\n")
    else:
        retrieval_clip = RetrievalCLIP(
            clip_architecture=args.architecture,
            nr_pred_frames=args.nr_pred_frames,
            nr_context_frames=args.nr_context_frames,
            tca_settings=tca_settings,
            vca_settings=vca_settings,
            optimizer=args.optimizer,
            permutation_mode=args.permutation_mode
        )

    clip_model, _ = clip.load(args.architecture, device='cpu')
    retrieval_clip.clip_model = clip_model

    test_trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=args.precision,
    )

    test_trainer.test(
        model=retrieval_clip,
        datamodule=datamodule,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='msrvtt', type=str)
    parser.add_argument('--data_root',
                        default='/Users/jochemloedeman/Documents/scriptie/data',
                        type=str)
    parser.add_argument('--ckpt_file_name',
                        default='video_6_1_adam.ckpt',
                        type=str)
    parser.add_argument('--architecture', default='ViT-B/32', type=str)
    parser.add_argument('--train_batch_size', default=2, type=int)
    parser.add_argument('--val_batch_size', default=2, type=int)
    parser.add_argument('--nr_context_frames', default=9, type=int)
    parser.add_argument('--nr_pred_frames', default=1, type=int)
    parser.add_argument('--video_resolution', default=112, type=int)
    parser.add_argument('--fps', default=9, type=int)
    parser.add_argument('--precision', default=32, type=int)
    parser.add_argument('--video_input_type', default="diff", type=str)
    parser.add_argument('--prompt_prefix', default="A video of ", type=str)
    parser.add_argument('--vca_length', default=6, type=int)
    parser.add_argument('--vca_mode', default='video', type=str)
    parser.add_argument('--image_sample_mode', default='center', type=str)
    parser.add_argument('--tca_length', default=2, type=int)
    parser.add_argument('--tca_mode', default='lm', type=str)
    parser.add_argument('--tca_insertion', default='infix', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--permutation_mode', default=None, type=str)

    parser.add_argument('--from_checkpoint',
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--dev_run', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--profiler', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--disable_vca', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--disable_tca', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--temporal_permutation',
                        action=argparse.BooleanOptionalAction,
                        default=False)

    args = parser.parse_args()

    main(args)
