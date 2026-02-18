import click
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

@click.command()
@click.option("--configs-dir", default="configs/train/vehicles", help="Directory containing train configs", show_default=True)
@click.option("--project", default="configs/project.yaml", help="Path to project config", show_default=True)
@click.option("--continue-on-error", is_flag=True, help="Continue to next config if one fails")
@click.option("--skip", multiple=True, help="Config names to skip")
@click.option("--pattern", default="*.yaml", help="Glob pattern for config files", show_default=True)
def main(configs_dir, project, continue_on_error, skip, pattern):
    load_dotenv()
    
    config_path = Path(configs_dir)
    if not config_path.exists():
        click.echo(f"Error: Directory {configs_dir} does not exist.")
        sys.exit(1)

    # Get all .yaml files in the directory
    configs = sorted(list(config_path.glob(pattern)))
    
    if not configs:
        click.echo(f"No YAML files found in {configs_dir}")
        sys.exit(0)

    click.echo(f"Found {len(configs)} configurations to run.")
    
    failed_configs = []

    for i, cfg in enumerate(configs, 1):
        if cfg.name in skip or cfg.stem in skip:
            click.echo(f"Skipping {cfg.name} as requested.")
            continue
            
        click.echo(f"\n[{i}/{len(configs)}] Running config: {cfg.name}")
        
        # Determine data config based on filename convention or content
        # Simplistic approach: parse YAML to find 'data' key, or infer from name
        # Since we put 'data: path/to/yaml' in the config files, we need to extract it
        # However, our train.py expects --data argument.
        # Let's read the config to extract the data path.
        
        try:
            # We can use a simple grep or read file to find "data: ..." line
            # or better, use pyyaml to parse it safely
            import yaml
            content = yaml.safe_load(cfg.read_text())

            cmd = [
                sys.executable, "scripts/train.py",
                "--project", project,
                "--train", str(cfg)
            ]
            
            click.echo(f"Executing: {' '.join(cmd)}")
            subprocess.check_call(cmd)
            
        except subprocess.CalledProcessError as e:
            click.echo(f"Error running {cfg.name}: {e}")
            failed_configs.append(cfg.name)
            if not continue_on_error:
                sys.exit(1)
        except Exception as e:
            click.echo(f"Unexpected error processing {cfg.name}: {e}")
            failed_configs.append(cfg.name)
            if not continue_on_error:
                sys.exit(1)

    if failed_configs:
        click.echo(f"\nSequence completed with failures in: {', '.join(failed_configs)}")
        sys.exit(1)
    else:
        click.echo("\nSequence completed successfully!")

if __name__ == "__main__":
    main()
