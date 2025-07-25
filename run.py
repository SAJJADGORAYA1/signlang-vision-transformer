from train import *

def main():
    dataset = SignDataset(VIDEO_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SignTransformer(
        input_dim=KEYPOINT_DIM,
        model_dim=MODEL_DIM,
        num_classes=len(set(dataset.encoded_labels)),
        h=NUM_HEADS,
        d_k=MODEL_DIM // NUM_HEADS,
        d_v=MODEL_DIM // NUM_HEADS,
        d_ff=MODEL_DIM * 4,
        n_layers=NUM_LAYERS,
        dropout_rate=0.1
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        train_model(model, dataloader, optimizer, criterion, EPOCHS)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    
    print("\nEvaluating on training data...")
    test_all_videos(model, dataset)
    
    print("\nTesting on single test video:")
    test_single_video(model, SINGLE_TEST_VIDEO, dataset.encoder)

if __name__ == "__main__":
    main()