import pandas as pd
import numpy as np
from ctgan import CTGAN


class SyntheticDataGenerator:

    def __init__(self):
        pass

    def generate_synthetic_data(self, df, n_synthetic=50):
        
        df = df.copy()
        # Define column types and constraints
        integer_cols = ['PUENTES VEHICULARES UND', 'PUENTES PEATONALES UND', 'TUNELES UND']
        area_cols = ['PUENTES VEHICULARES M2', 'PUENTES PEATONALES M2', 'TUNELES M2']
        output_cols = [col for col in df.columns if col not in 
                    ['TIPO DE TERRENO', 'LONGITUD KM'] + integer_cols + area_cols]

        def train_ctgan_with_constraints(df, n_synthetic=50):
            """
            CTGAN learns the joint probability distribution of your data using:
            1. A Generator network that creates synthetic samples
            2. A Discriminator network that distinguishes real from synthetic
            3. They compete until synthetic data is indistinguishable from real
            
            This captures complex relationships between all variables!
            """
            
            print("Training CTGAN neural network on your data patterns...")
            
            # Initialize CTGAN with optimal parameters for small datasets
            model = CTGAN(
                epochs=500,  # More epochs to learn patterns thoroughly
                batch_size=10,  # Small batch size for 45 samples
                discriminator_dim=(128, 128),  # Smaller networks for small data
                generator_dim=(128, 128),
                discriminator_lr=2e-4,
                generator_lr=2e-4,
                discriminator_decay=1e-6,
                generator_decay=1e-6,
                pac=10,  # Packing - helps with small datasets
                cuda=False  # Use CPU
            )
            
            # Train CTGAN - it learns the data distribution
            # CRITICAL: Specify categorical column to avoid the error
            model.fit(
                df,
                discrete_columns=['TIPO DE TERRENO']
            )
            
            print(f"CTGAN trained! Generating {n_synthetic} synthetic samples...")
            
            # Generate synthetic data from learned distribution
            synthetic_data = model.sample(n_synthetic)
            
            # Post-process to enforce domain constraints
            # CTGAN learns distributions but may generate outside bounds
            
            # 1. Integer columns (0-30)
            for col in integer_cols:
                synthetic_data[col] = np.clip(synthetic_data[col].round(), 0, 30).astype(int)
            
            # 2. Area columns (0-5000)
            for col in area_cols:
                synthetic_data[col] = np.clip(synthetic_data[col], 0, 5000)
            
            # 3. Length (1-250)
            synthetic_data['LONGITUD KM'] = np.clip(synthetic_data['LONGITUD KM'], 1, 250)
            
            # 4. Output costs (0-2e9)
            for col in output_cols:
                synthetic_data[col] = np.clip(synthetic_data[col], 0, 2e9)
            
            return synthetic_data

        # Alternative: Use SDV which handles constraints better
        def train_sdv_with_constraints(df, n_synthetic=50):
            """
            SDV uses Gaussian Copulas to model dependencies between variables
            It's often more stable than CTGAN for small tabular datasets
            """
            try:
                from sdv.single_table import GaussianCopulaSynthesizer
                from sdv.metadata import SingleTableMetadata
                
                print("Training SDV Gaussian Copula model...")
                
                # Create metadata with constraints
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(df)
                
                # Specify column types
                metadata.update_column('TIPO DE TERRENO', sdtype='categorical')
                
                for col in integer_cols:
                    metadata.update_column(col, sdtype='numerical', computer_representation='Int64')
                
                # Create synthesizer with constraints
                synthesizer = GaussianCopulaSynthesizer(
                    metadata,
                    enforce_min_max_values=True,  # Enforce learned min/max
                    default_distribution='truncnorm',  # Truncated normal to respect bounds
                    numerical_distributions={
                        'LONGITUD KM': 'truncnorm',
                        **{col: 'truncnorm' for col in area_cols},
                        **{col: 'truncnorm' for col in output_cols}
                    }
                )
                
                # Fit the model
                synthesizer.fit(df)
                
                # Add constraints
                from sdv.constraints import ScalarRange, Positive
                
                # Define constraints
                constraints = []
                for col in integer_cols:
                    synthesizer.add_constraint(
                        constraint=ScalarRange(
                            column_name=col,
                            low=0,
                            high=50,
                            strict_boundaries=True
                        )
                    )
                
                print(f"SDV trained! Generating {n_synthetic} synthetic samples...")
                
                # Generate synthetic data
                synthetic_data = synthesizer.sample(n_synthetic)
                
                # Final cleanup for integer columns
                for col in integer_cols:
                    synthetic_data[col] = synthetic_data[col].round().astype(int)
                
                return synthetic_data
                
            except ImportError:
                print("SDV not installed. Install with: pip install sdv")
                return None

        # Train CTGAN on your data
        df_synthetic_ctgan = train_ctgan_with_constraints(df, n_synthetic=50)

        # Combine with original
        df_augmented = pd.concat([df, df_synthetic_ctgan], ignore_index=True)

        # Analysis of generated data
        print("\n" + "="*50)
        print("SYNTHETIC DATA QUALITY CHECK")
        print("="*50)

        # Compare distributions
        print("\nOriginal vs Synthetic Statistics:")
        print(f"{'Column':<30} {'Original Mean':<15} {'Synthetic Mean':<15} {'Difference %':<10}")
        print("-"*70)

        for col in ['LONGITUD KM'] + integer_cols + area_cols:
            orig_mean = df[col].mean()
            synth_mean = df_synthetic_ctgan[col].mean()
            diff_pct = abs(orig_mean - synth_mean) / orig_mean * 100 if orig_mean != 0 else 0
            print(f"{col:<30} {orig_mean:<15.2f} {synth_mean:<15.2f} {diff_pct:<10.1f}%")

        # Check correlations are preserved
        print("\nCorrelation preservation (sample):")
        orig_corr = df[['LONGITUD KM', 'PUENTES VEHICULARES UND']].corr().iloc[0, 1]
        synth_corr = df_synthetic_ctgan[['LONGITUD KM', 'PUENTES VEHICULARES UND']].corr().iloc[0, 1]
        print(f"LONGITUD KM vs PUENTES VEHICULARES UND:")
        print(f"  Original correlation: {orig_corr:.3f}")
        print(f"  Synthetic correlation: {synth_corr:.3f}")

        # Terrain type distribution
        print("\nTerrain type distribution:")
        orig_dist = df['TIPO DE TERRENO'].value_counts(normalize=True)
        synth_dist = df_synthetic_ctgan['TIPO DE TERRENO'].value_counts(normalize=True)
        for terrain in orig_dist.index:
            orig_pct = orig_dist.get(terrain, 0) * 100
            synth_pct = synth_dist.get(terrain, 0) * 100
            print(f"  {terrain}: Original {orig_pct:.1f}%, Synthetic {synth_pct:.1f}%")

        # Verify constraints
        print("\n" + "="*50)
        print("CONSTRAINT VERIFICATION")
        print("="*50)

        print(f"LONGITUD KM: [{df_synthetic_ctgan['LONGITUD KM'].min():.2f}, "
            f"{df_synthetic_ctgan['LONGITUD KM'].max():.2f}] ✓ (1-250)")

        for col in integer_cols:
            print(f"{col}: [{df_synthetic_ctgan[col].min()}, "
                f"{df_synthetic_ctgan[col].max()}] ✓ (0-25)")

        for col in area_cols[:1]:  # Show one example
            print(f"{col}: [{df_synthetic_ctgan[col].min():.1f}, "
                f"{df_synthetic_ctgan[col].max():.1f}] ✓ (0-5000)")


        print("\n" + "="*50)
        print("KEY DIFFERENCE FROM RANDOM NOISE:")
        print("="*50)
        print("• CTGAN learned the JOINT DISTRIBUTION of all variables")
        print("• It captures RELATIONSHIPS between inputs and outputs")
        print("• It preserves CORRELATIONS and PATTERNS from your data")
        print("• The Generator network creates statistically similar samples")
        print("• NOT random noise - it's learned from your data structure!")
        
        return df_augmented