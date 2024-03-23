class Utils:
    @staticmethod
    def version_check(package, requirements):
        """
        Checks if the installed version of a package matches the required version specified in requirements.txt.

        Args:
            package (module): The package/module whose version needs to be checked.
            requirements (list): A list of strings representing the requirements.txt file content.

        Returns:
            bool: True if the installed version matches the required version, False otherwise.
                  Returns True if requirements.txt file is not found.
                  Returns False if version information for the package is not found in requirements.txt.
        Raises:
            None
        """

        bullet = ">>>"

        if not requirements:
            print(f"{bullet} Error: requirements.txt is empty.")
            return False

        required_version = None

        for line in requirements:
            if package.__name__ in line.lower():
                required_version = line.split("==")[1].strip()
                break

        if required_version is None:
            print(
                f"{bullet} Error: Could not find version information for {package.__name__} in requirements.txt. Skipping..."
            )
            return False

        if package.__version__ != required_version:
            print(f"Error: {package.__name__} version mismatch:")
            print(f"  -> Require {package.__name__} {required_version}.")
            print(f"  -> Current {package.__name__} {package.__version__}.")
            print(f"{bullet} Please update to the required version: {required_version}")
            return False
        else:
            print(f"{bullet} {package.__name__} is up to date: {required_version}")
            return True
