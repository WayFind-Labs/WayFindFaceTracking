import argparse
import sys
import os
import json
from arcface_service import ArcFaceService

def main():
    parser = argparse.ArgumentParser(
        description="ArcFace Vector Tool: Convert images to vectors, save them, and compare vectors."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command 1: Convert image to vector
    convert_parser = subparsers.add_parser(
        "convert", 
        help="Convert an image to an ArcFace 512-D vector and save it."
    )
    convert_parser.add_argument("--image", "-i", required=True, help="Path to input image file")
    convert_parser.add_argument("--save", "-s", help="Path to save output vector (.json or .npy)")
    convert_parser.add_argument("--name", "-n", help="Optional person name metadata")

    # Command 2: Compare two vectors
    compare_parser = subparsers.add_parser(
        "compare", 
        help="Compare two face vectors (from files or images)."
    )
    compare_parser.add_argument("--vec1", "-v1", required=True, help="Path to first vector (.json or .npy) or image file")
    compare_parser.add_argument("--vec2", "-v2", required=True, help="Path to second vector (.json or .npy) or image file")
    compare_parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Similarity threshold (default: 0.5)")

    args = parser.parse_args()

    if args.command == "convert":
        print(f"[Convert] Processing image: {args.image}")
        service = ArcFaceService()
        metadata = {"person_name": args.name} if args.name else None
        
        try:
            vector = service.image_to_vector(args.image, save_path=args.save, metadata=metadata)
            print(f"[Success] Vector generated (Dimension: {len(vector)})")
            if args.save:
                print(f"[Success] Saved to: {os.path.abspath(args.save)}")
            else:
                print(f"Vector preview (first 5 elements): {vector[:5].tolist()}")
        except Exception as e:
            print(f"[Error] Failed to convert image to vector: {e}")
            sys.exit(1)

    elif args.command == "compare":
        print(f"[Compare] Vector 1: {args.vec1}")
        print(f"[Compare] Vector 2: {args.vec2}")
        
        # Check if inputs are image paths or vector files
        service = None
        v1_input = args.vec1
        v2_input = args.vec2

        ext1 = os.path.splitext(args.vec1)[1].lower()
        ext2 = os.path.splitext(args.vec2)[1].lower()

        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        if ext1 in image_exts:
            service = ArcFaceService()
            v1_input = service.image_to_vector(args.vec1)
        if ext2 in image_exts:
            if service is None:
                service = ArcFaceService()
            v2_input = service.image_to_vector(args.vec2)

        try:
            result = ArcFaceService.compare_vectors(v1_input, v2_input, threshold=args.threshold)
            print("\n=== Vector Comparison Result ===")
            print(json.dumps(result, indent=2))
            
            if result["is_match"]:
                print(f"\n verdict: MATCH (Similarity {result['similarity_score']} >= Threshold {result['threshold']})")
            else:
                print(f"\n verdict: NO MATCH (Similarity {result['similarity_score']} < Threshold {result['threshold']})")
                
        except Exception as e:
            print(f"[Error] Failed to compare vectors: {e}")
            sys.exit(1)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
