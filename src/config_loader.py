import yaml
import re

class Config:
    def __init__(self, path):
        """
        Initialize Config class with parameters from a YAML file.
        
        Args:
            path (str): Path to the YAML configuration file.
        """
        self._config = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {path}")
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file: {exc}")

        # Flatten the configuration and set as class attributes
        if self._config:
            for section, params in self._config.items():
                if isinstance(params, dict):
                    for key, value in params.items():
                        setattr(self, key, self._parse_value(value))
                else:
                    # Handle top-level keys if any
                    setattr(self, section, self._parse_value(params))

    def _parse_value(self, value):
        """
        Parse value to float if it looks like scientific notation but was loaded as string.
        PyYAML 1.2 loader might load '1e-4' as string.
        """
        if isinstance(value, str):
            # Check for scientific notation (e.g., "1e-4", "5E-5", "-1.2e3")
            # Simple regex: optional sign, digits, optional dot+digits, e/E, optional sign, digits
            # But we only care about the cases PyYAML missed, which are usually simple like 1e-4
            if re.match(r'^-?\d+(\.\d+)?[eE][-+]?\d+$', value):
                try:
                    return float(value)
                except ValueError:
                    pass
        return value

    def __repr__(self):
        """
        Return a string representation of the configuration.
        """
        # Filter out private attributes and methods
        params = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"Config({params})"
