gpu_cmd="$(lspci -mm | awk -F '\"|\" \"|\\(' \
    '/"Display|"3D|"VGA/ {
        a[$0] = $1 " " $3 " " ($(NF-1) ~ /^$|^Device [[:xdigit:]]+$/ ? $4 : $(NF-1))
    }
    END { for (i in a) {
        if (!seen[a[i]]++) {
            sub("^[^ ]+ ", "", a[i]);
            print a[i]
        }
    }}')"

IFS=$'\n' read -d "" -ra gpus <<< "$gpu_cmd"

# Remove duplicate Intel Graphics outputs.
# This fixes cases where the outputs are both
# Intel but not entirely identical.
#
# Checking the first two array elements should
# be safe since there won't be 2 intel outputs if
# there's a dedicated GPU in play.
[[ "${gpus[0]}" == *Intel* && "${gpus[1]}" == *Intel* ]] && unset -v "gpus[0]"

for gpu in "${gpus[@]}"; do
    # GPU shorthand tests.
    [[ "$gpu_type" == "dedicated" && "$gpu" == *Intel* ]] || \
    [[ "$gpu_type" == "integrated" && ! "$gpu" == *Intel* ]] && \
        { unset -v gpu; continue; }

    case $gpu in
        *"Advanced"*)
            brand="${gpu/*AMD*ATI*/AMD ATI}"
            brand="${brand:-${gpu/*AMD*/AMD}}"
            brand="${brand:-${gpu/*ATI*/ATi}}"

            gpu="${gpu/\[AMD\/ATI\] }"
            gpu="${gpu/\[AMD\] }"
            gpu="${gpu/OEM }"
            gpu="${gpu/Advanced Micro Devices, Inc.}"
            gpu="${gpu/*\[}"
            gpu="${gpu/\]*}"
            gpu="$brand $gpu"
        ;;

        *"NVIDIA"*)
            gpu="${gpu/*\[}"
            gpu="${gpu/\]*}"
            gpu="NVIDIA $gpu"
        ;;

        *"Intel"*)
            gpu="${gpu/*Intel/Intel}"
            gpu="${gpu/\(R\)}"
            gpu="${gpu/Corporation}"
            gpu="${gpu/ \(*}"
            gpu="${gpu/Integrated Graphics Controller}"
            gpu="${gpu/*Xeon*/Intel HD Graphics}"

            [[ -z "$(trim "$gpu")" ]] && gpu="Intel Integrated Graphics"
        ;;

        *"MCST"*)
            gpu="${gpu/*MCST*MGA2*/MCST MGA2}"
        ;;

        *"VirtualBox"*)
            gpu="VirtualBox Graphics Adapter"
        ;;

        *) continue ;;
    esac

    touch enum-gpu
    file_data=($(wc enum-gpu))
    line_count=${file_data[0]}

    echo "$gpu" >> enum-gpu

done
