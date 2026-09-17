
#!/bin/bash

directory=$(pwd)
echo "Current dir: ${directory}"
echo "Update or Synchronize Conda Env?"
select choice in "Update" "Sync" "Exit"; do
    case $choice in
        "Update")
            echo "Updating YAML"
	    conda env export > conda_env.yml
            break  # Exit the loop after selection
            ;;
        "Sync")
            echo "Syncing"
	    conda env update --file conda_env.yml  --prune
            break
            ;;
        "Exit")
            echo "Exiting... Goodbye! 👋"
            break
            ;;
        *)  # Handle invalid input
            echo "Invalid option. Please enter a number (1-3)."
            ;;
    esac
done
