.PHONY: clean All

All:
	@echo "----------Building project:[ SOM - Debug ]----------"
	@cd "SOM" && "$(MAKE)" -f  "SOM.mk"
clean:
	@echo "----------Cleaning project:[ SOM - Debug ]----------"
	@cd "SOM" && "$(MAKE)" -f  "SOM.mk" clean
